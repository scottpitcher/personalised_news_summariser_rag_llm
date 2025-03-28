import os
import requests
from dotenv import load_dotenv
from datetime import datetime,timedelta
import json
import time

# Set up 
load_dotenv(override=False)
API_KEY = os.getenv("NEWS_API_KEY")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Config
CATEGORIES = ["technology", "health", "business", "science", "general"]
PAGE_SIZE = 100
TODAY = datetime.now().strftime("%Y-%m-%d")
YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
FROM = (datetime.utcnow() - timedelta(hours=24)).isoformat(timespec="seconds")
TO = datetime.utcnow().isoformat(timespec="seconds")


def fetch_articles_categories(category="technology", page_size=10):
    # Only get top headlines
    url = "https://newsapi.org/v2/top-headlines"

    # Other article specifications
    params = {
        "category": category,
        "language": "en",
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    # API call
    response = requests.get(url, params=params)
    response.raise_for_status()
    articles = response.json().get("articles", [])

    parsed = []
    for a in articles:
        content = a.get("content") or a.get("description") or ""
        if not content:
            continue
        parsed.append({
            "title": a.get("title", "").strip(),
            "content": content.strip(),
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "category": category
        })
    return parsed

def fetch_politics_data(page_size=10):
    # Get from the everything page for custom querying for Politics
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "politics OR congress OR government OR election OR white house OR Trump OR Vance OR washington",
        "language": "en",
        "sortBy": "popularity",
        "from":YESTERDAY,
        "pageSize": page_size,
        "apiKey": API_KEY
    }

    # API call
    response = requests.get(url, params=params)
    response.raise_for_status()
    articles = response.json().get("articles", [])

    parsed = []
    for a in articles:
        content = a.get("content") or a.get("description") or ""
        if not content:
            continue
        parsed.append({
            "title": a.get("title", "").strip(),
            "content": content.strip(),
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "category": "politics"
        })
    return parsed


if __name__ == "__main__":
    os.makedirs("data/articles", exist_ok=True)
    total_articles = 0

    for category in CATEGORIES:
        data = fetch_articles_categories(category, page_size= PAGE_SIZE)
        filename = f"data/articles/{TODAY}_{category}.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} {category} articles to {filename}")
        total_articles += len(data)
        time.sleep(1.1) 
    
    # Custom "politics" category
    politics_data = fetch_politics_data(page_size=PAGE_SIZE)
    politics_filename = f"data/articles/{TODAY}_politics.json"
    with open(politics_filename, "w") as f:
        json.dump(politics_data, f, indent=2)
    print(f"Saved {len(politics_data)} politics articles to {politics_filename}")
    total_articles += len(politics_data)

    # Placeholder for other custom custom categories
    ################################################

    print(f"\nTotal articles saved: {total_articles}")
