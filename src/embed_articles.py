import os
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Setup
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL_TEXT_DIR = Path("data/full_text")
VECTOR_STORE_DIR = Path("data/vector_store")
MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_PATH = VECTOR_STORE_DIR / "global_index.faiss"
META_PATH = VECTOR_STORE_DIR / "global_metadata.pkl"
SEEN_PATH = VECTOR_STORE_DIR / "seen_articles.json"

print("Importing model...")
model = SentenceTransformer(MODEL_NAME)

def load_seen_urls():
    print("Loading seen files...")
    if SEEN_PATH.exists() and SEEN_PATH.stat().st_size > 0:
        with open(SEEN_PATH, "r") as f:
            return set(json.load(f).get("embedded_urls", []))
    return set()


def save_seen_urls(urls):
    print("Saving seen files...")
    with open(SEEN_PATH, "w") as f:
        json.dump({"embedded_urls": list(urls)}, f, indent=2)

def load_existing_index_and_metadata():
    print("Importing FAISS file...")
    if INDEX_PATH.exists() and META_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, []

def embed_new_articles():
    print("Searching files...")
    all_files = sorted(FULL_TEXT_DIR.glob("*.json"))
    seen_urls = load_seen_urls()
    index, metadata = load_existing_index_and_metadata()

    new_texts = []
    new_metadata = []

    print("Filtering out seen files...")
    for file in all_files:
        with open(file, "r") as f:
            articles = json.load(f)

        for article in articles:
            url = article.get("url")
            content = article.get("content", "").strip()
            if not url or not content or url in seen_urls:
                continue

            new_texts.append(content)
            new_metadata.append({
                "title": article.get("title", ""),
                "url": url,
                "source": article.get("source", ""),
                "category": article.get("category", ""),
                "date": file.stem  # e.g. "2025-03-25"
            })
            seen_urls.add(url)

    if not new_texts:
        print("No new articles to embed.")
        return

    print(f"Embedding {len(new_texts)} new articles...")
    embeddings = model.encode(new_texts, show_progress_bar=True)

    # Build or update index
    dim = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Extend and save metadata
    metadata.extend(new_metadata)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    save_seen_urls(seen_urls)

    print(f"Updated global index with {len(new_texts)} articles.")

if __name__ == "__main__":
    embed_new_articles()
