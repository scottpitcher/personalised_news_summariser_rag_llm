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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SUMMARIZER_MODEL_NAME = "facebook/bart-base" 
local_summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
local_summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)


# Config
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_STORE_DIR = Path("data/vector_store")
FULL_TEXT_DIR = Path("data/full_text")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
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
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

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

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.7,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def summarize_articles_local(query, articles_text):
    prompt = (
        f"You are a helpful assistant summarizing news for a user. "
        f"User asked: \"{query}\". "
        f"Based on the following articles, provide a concise summary:\n\n"
        f"{articles_text}"
    )
    inputs = local_summarizer_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    summary_ids = local_summarizer_model.generate(inputs, max_length=500, min_length=50, do_sample=True)
    summary = local_summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main query pipeline
def query_news(summarizer_model, query):
    # Embed the query using your existing embedding model (assumed to be loaded as 'model')
    query_embedding = embedding_model.encode([query])

    # D: Query-Embedding distances; I: Indices of matched vectors from your FAISS index
    D, I = index.search(query_embedding, TOP_K)

    selected_texts = []
    match_summaries = []  # store printable match strings
    print("\nTop Matches:\n" + "-" * 40)

    for i, score in zip(I[0], D[0]):
        article = metadata[i]
        title = article['title']
        source = article['source']
        score_str = f"{score:.4f}"

        content = get_article_content(article["url"], article["date"])
        status = "[KEPT]" if score <= RELEVANCE_THRESHOLD and content else "[OMITTED]"

        match_str = f"""📌 {title} {status}
        • Source: {source}, {article['date']}
        • Relevance Score: {score_str}"""
        
        print(match_str)
        match_summaries.append(match_str)
        
        if score <= RELEVANCE_THRESHOLD and content:
            selected_texts.append(content)

    if not selected_texts:
        print("No matching content found.")
        return {
            "summary": "I'm sorry, I couldn't find any relevant articles.",
            "matches": match_summaries
        }

    # Combine all selected article texts into one large string
    all_text = "\n\n---\n\n".join(selected_texts)

    # Choose the summarization method based on the summarizer_model parameter
    if summarizer_model == 'openai':
        summary = summarize_articles(query, all_text)
    elif summarizer_model == 'huggingface':
        # Call your local summarization function implemented with a HuggingFace model.
        summary = summarize_articles_local(query, all_text)
    else:
        raise ValueError("Unsupported summarizer_model. Choose 'openai' or 'huggingface'.")

    print("\nSummary:\n" + "-" * 40)
    print(summary)

    return {
        "summary": summary or "Sorry, no summary could be generated.",
        "matches": match_summaries
    }


# Run
if __name__ == "__main__":
    user_query = input("What would you like to know about? ")
    query_news(summarizer_model = 'openai', query = user_query)