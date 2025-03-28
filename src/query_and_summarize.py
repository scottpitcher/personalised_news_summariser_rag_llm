# query_and_summarize.py

# The focus of this script is to  
## Embed a user query,
## Search the FAISS vector store,
## Retrieve the top article contents,
## Send them to an LLM for summarization
import os
import faiss
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(override=False)

# Config
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_STORE_DIR = Path("data/vector_store")
FULL_TEXT_DIR = Path("data/full_text")
MODEL_NAME = "all-MiniLM-L6-v2" # Model for embedding
RELEVANCE_THRESHOLD = 1.2       # controls relevance of chosen articles
TOP_K = 3                       # Number of top articles to retrieve

# Set up openai
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load vector index and metadata
index = faiss.read_index(str(VECTOR_STORE_DIR / "global_index.faiss"))
with open(VECTOR_STORE_DIR / "global_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Lookup article content by URL + date
def get_article_content(url, date):
    file_path = FULL_TEXT_DIR / f"{date}.json"
    if not file_path.exists():
        return None

    with open(file_path, "r") as f:
        articles = json.load(f)
        for article in articles:
            if article.get("url") == url:
                return article.get("content")
    return None

# Generate summary using OpenAI
def summarize_articles(query, articles_text):
    prompt = f"""
You are a helpful assistant summarizing current news for a user.
It is important the response only pertains to the user's query.
If there is not enough information to answer accurately, indicate a lack of knowledge. 

User asked: "{query}"

Based on the following articles, provide a clear, casual answer to the query.:

{articles_text}
    """.strip()

    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()

# Main query pipeline
def query_news(query):
    query_embedding = model.encode([query])

    # D: Query-Embedding distance; I: Indices of matched vectors
    D, I = index.search(query_embedding, TOP_K)

    selected_texts = []
    match_summaries = []  # store printable match strings
    print("\nTop Matches:\n" + "-"*40)

    for i, score in zip(I[0], D[0]):
        article = metadata[i]
        title = article['title']
        source = article['source']
        score_str = f"{score:.4f}"

        match_str = f"📌 {title} ({source}) — Score: {score_str}"
        print(match_str)
        match_summaries.append(match_str)

        content = get_article_content(article["url"], article["date"])
        if content:
            selected_texts.append(content)

    if not selected_texts:
        print("No matching content found.")
        return {
            "summary": "I'm sorry, I couldn't find any relevant articles.",
            "matches": match_summaries
        }

    all_text = "\n\n---\n\n".join(selected_texts)
    summary = summarize_articles(query, all_text)

    print("\nSummary:\n" + "-"*40)
    print(summary)

    return {
        "answer": summary or "Sorry, no summary could be generated.",
        "matches": match_summaries
    }


# Run
if __name__ == "__main__":
    user_query = input("What would you like to know about? ")
    query_news(user_query)