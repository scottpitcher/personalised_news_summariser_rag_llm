# run_daily_news_pipeline.py
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project structure
from fetch_urls import fetch_articles_categories, fetch_politics_data, CATEGORIES, PAGE_SIZE
from scrape_full_articles import process_articles_for_date, check_unscraped_urls
from embed_articles import embed_new_articles
from query_and_summarize import query_news

# Config
TODAY = datetime.now().strftime("%Y-%m-%d")
ARTICLES_DIR = Path("data/articles")
FULL_TEXT_DIR = Path("data/full_text")

ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
FULL_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: Fetch all categories + custom 'politics'
def fetch_all_categories():
    total = 0
    for category in CATEGORIES:
        print(f"Fetching {category} articles...")
        data = fetch_articles_categories(category, page_size=PAGE_SIZE)
        if data:
            filename = ARTICLES_DIR / f"{TODAY}_{category}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved {len(data)} {category} articles")
            total += len(data)

    print("📥 Fetching custom 'politics' articles...")
    politics_data = fetch_politics_data(page_size=30)
    if politics_data:
        filename = ARTICLES_DIR / f"{TODAY}_politics.json"
        with open(filename, "w") as f:
            json.dump(politics_data, f, indent=2)
        print(f"✅ Saved {len(politics_data)} politics articles")
        total += len(politics_data)

    print(f"\nTotal articles fetched: {total}")
    return total

# Step 2: Scrape full content
def scrape_all():
    files_by_date = check_unscraped_urls()
    if not files_by_date:
        print("✅ All dates already have full text data!")
    else:
        for date, input_files in files_by_date.items():
            process_articles_for_date(date, input_files)

# Step 3: Embed new articles
def embed_all():
    print("📌 Embedding articles...")
    embed_new_articles()

# Step 4: Ask user for a query
def handle_query():
    query = input("\n🔍 Ask a question about today’s news: ")
    if query:
        query_news(query)

# Run full pipeline
if __name__ == "__main__":
    print("🔄 Running full news summarization pipeline...\n")
    fetch_all_categories()
    scrape