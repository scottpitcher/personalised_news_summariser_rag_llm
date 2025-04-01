from query_and_summarize import query_news
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    user_query = input("🔍 Ask a question about today’s news: ")
    query_news(user_query)
