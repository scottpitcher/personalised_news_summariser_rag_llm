import os
import json
from datetime import datetime
from pathlib import Path
from query_and_summarize import query_news

# Define the path for the feedback log (JSON file).
FEEDBACK_LOG_PATH = Path("data/fine_tune_data/feedback_log.json")

def log_feedback(query, summary, feedback):
    """
    Log feedback into a JSON file. Each entry includes the query,
    the generated summary, the user's feedback, and a timestamp.
    """
    entry = {
        "query": query,
        "summary": summary,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Load existing log entries if the file exists; otherwise, create a new list.
    if FEEDBACK_LOG_PATH.exists():
        try:
            with open(FEEDBACK_LOG_PATH, "r") as f:
                log_entries = json.load(f)
        except json.JSONDecodeError:
            log_entries = []
    else:
        log_entries = []

    log_entries.append(entry)

    # Write updated log back to the JSON file.
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(log_entries, f, indent=2)

def get_valid_feedback():
    """
    Repeatedly prompt the user until valid feedback is provided.
    Accepts:
      - "y" or "yes" for thumbs up.
      - "n" or "no" for thumbs down.
      - A rewrite suggestion with at least 10 characters.
    """
    while True:
        feedback_input = input("Enter your feedback (y / n / rewrite text): ").strip().lower()
        if feedback_input in ["y", "yes"]:
            return "thumbs_up"
        elif feedback_input in ["n", "no"]:
            return "thumbs_down"
        elif len(feedback_input) >= 10:
            return {"rewrite": feedback_input}
        else:
            print("Invalid input. Please enter 'y/yes', 'n/no', or a rewrite suggestion with at least 10 characters.")

def main():
    # Step 1: Accept user query.
    query = input("What would you like to know about? ")

    # Step 2 & 3: Call existing RAG pipeline to retrieve summary.
    result = query_news(model = 'huggingface', query= query)
    summary = result.get("summary")
    
    print("\nSummary:\n" + "-" * 40)
    print(summary)
    
    # Step 4: Collect and validate user feedback.
    feedback = get_valid_feedback()
    
    # Log the query, summary, and feedback.
    log_feedback(query, summary, feedback)
    
    print("Feedback logged.")

if __name__ == "__main__":
    main()