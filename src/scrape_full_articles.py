import json
import os
from collections import defaultdict
from newspaper import Article
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FIELDS_TO_KEEP = ["title", "category", "url", "source"]

def get_date_from_filename(filename):
    # Assumes format like: 2025-03-25_tech.json
    return filename.split("_")[0]

def check_unscraped_urls(articles_dir="data/articles", full_text_dir="data/full_text"):
    """
    Identify which dates have not yet been processed into full text files.
    """
    scraped_dates = {f.replace(".json", "") for f in os.listdir(full_text_dir)}
    dates_to_process = defaultdict(list)

    for file in os.listdir(articles_dir):
        if file.endswith(".json"):
            date = get_date_from_filename(file)
            if date not in scraped_dates:
                dates_to_process[date].append(os.path.join(articles_dir, file))

    return dates_to_process  # { "2025-03-25": [list of category files] }


def scrape_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"[!] Failed to scrape {url}: {e}")
        return None

def process_articles_for_date(date, input_files):
    seen_urls = set()
    all_articles = []

    for input_path in input_files:
        with open(input_path, "r") as f:
            raw_articles = json.load(f)

        for article in tqdm(raw_articles, desc=f"Scraping for {date} - {os.path.basename(input_path)}"):
            url = article.get("url", "")
            if url in seen_urls:
                continue
            content = scrape_article_content(url)
            if not content:
                continue
            cleaned = {k: article[k] for k in FIELDS_TO_KEEP if k in article}
            cleaned["content"] = content
            seen_urls.add(url)
            all_articles.append(cleaned)

    # Save combined full text file
    output_path = f"data/full_text/{date}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_articles, f, indent=2)

    print(f"\nSaved {len(all_articles)} total articles to: {output_path}")

if __name__ == "__main__":
    files_by_date = check_unscraped_urls()
    if not files_by_date:
        print("✅ All dates already have full text data!")
    else:
        for date, input_files in files_by_date.items():
            process_articles_for_date(date, input_files)