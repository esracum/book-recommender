import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

# ---(VERİ YÜKLEME VE FONKSİYONLAR) ---
load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "not-found.jpg",
    books["large_thumbnail"],
)



# Yerelde çalıştırırken her seferinde yeniden tokenizayon olmasın diye Chroma DB'yi kaydetmek için bir klasör yolu belirlendi

persist_directory = "D:\\bookRecommender\\chroma_db_data"  # Benim diskim yetmiyordu ondan D :)

# Eğer 'chroma_db_data' klasörü mevcutsa ve içinde dosya varsa, DB'yi diskten yükle
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Chroma DB zaten mevcut, diskten yükleniyor...")
    # Not: Embedding fonksiyonunu tekrar tanımlamak gerekir çünkü Chroma yüklerken bunu bekler.
    db_books = Chroma(
        persist_directory=persist_directory,
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    )
else:
    # Aksi takdirde, metinleri yükle, embedding'leri oluştur ve diske kaydet
    print("Chroma oluşturuluyor ve diske kaydediliyor...")
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    db_books = Chroma.from_documents(
        documents,
        GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        persist_directory=persist_directory
    )

    print(f"Chroma DB '{persist_directory}' klasörüne kaydedildi.")

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# --- ARAYÜZ KISMI ---

# 1. Özel CSS Tanımlaması
custom_css = """
#header-title {
    text-align: center;
    font-size: 1.8rem; /* Başlık boyutu */
    margin-bottom: 0.5rem;
    color: #10b981; /* Başlık rengi */
}
#header-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #7f8c8d;
    margin-bottom: 2rem;
}
.gradio-container {
    font-family: 'Helvetica Neue', sans-serif;
}
"""

# 2. Tema Seçimi
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    text_size="lg",
    spacing_size="md",
)

with gr.Blocks(theme=theme, css=custom_css, title="Book Recommender") as dashboard:
    # 3. HTML Başlık Alanı
    gr.HTML("""
        <h2 id="header-title"> AI BOOK RECOMMENDER </h2>
        <p id="header-subtitle">Find the perfect book for your mood using AI-powered semantic search.</p>
    """)

    with gr.Row():
        # Sol taraf: Arama ve Filtreler
        with gr.Column(scale=2, variant="panel"):
            user_query = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="E.g., An inspiring story about struggle ...",
                lines=1
            )

            with gr.Group():
                category_dropdown = gr.Dropdown(
                    choices=categories,
                    label="Category",
                    value="All",
                    interactive=True
                )
                tone_dropdown = gr.Dropdown(
                    choices=tones,
                    label="Emotional Tone",
                    value="All",
                    interactive=True,

                )

            submit_button = gr.Button("Find books", variant="primary", size="lg")



        # Sağ taraf: Sonuçlar
        with gr.Column(scale=3):
            gr.Markdown("### Featured Selections")
            output = gr.Gallery(
                label="Recommended Books",
                columns=[4],
                rows=[4],
                object_fit="contain",
                height="auto"
            )

    # Buton
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


if __name__ == "__main__":
    dashboard.launch()