# rlhf_finetune.py
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import PPOConfig, PPOTrainer

# Config & Paths
FEEDBACK_LOG_PATH = Path("data/fine_tune_data/feedback_log.json")
MODEL_NAME = "t5-base"   # or facebook/bart-base, etc.
DEVICE = torch.device("cpu")  # or "cuda" if you have a GPU


# Reward Mapping
def calculate_reward(feedback):
    """
    Map raw feedback to a numeric reward.
    thumbs_up   -> +1.0
    thumbs_down -> -1.0
    {"rewrite": "..."} -> -0.5
    """
    if isinstance(feedback, str):
        if feedback == "thumbs_up":
            return 1.0
        if feedback == "thumbs_down":
            return -1.0
    if isinstance(feedback, dict) and "rewrite" in feedback:
        return -0.5
    return 0.0

# Load Feedback Log
def load_feedback_data():
    """
    Reads the JSON feedback log and returns a list of
    (query, summary, reward) tuples for RL training.
    """
    if not FEEDBACK_LOG_PATH.exists():
        return []

    with open(FEEDBACK_LOG_PATH, "r") as f:
        entries = json.load(f)

    dataset = []
    for e in entries:
        q = e.get("query", "")
        s = e.get("summary", "")
        r = calculate_reward(e.get("feedback"))
        dataset.append((q, s, r))
    return dataset

# Setup Model & PPO
# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Configure PPO
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1e-5,
    batch_size=2,        # adjust to your GPU/memory
    ppo_epochs=4,
    optimize_cuda_cache=True
)

# Instantiate the PPO trainer
ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

# RLHF Training Loop
def rlhf_finetune(dataset, iterations=1):
    """
    Runs RL fine‑tuning over the dataset for a given
    number of full passes (iterations).
    """
    for it in range(iterations):
        print(f"=== RLHF Iteration {it+1}/{iterations} ===")
        for query, summary, reward in dataset:
            # PPO expects lists of queries & generations
            queries    = [query]
            generations = [summary]
            rewards    = [reward]

            # Perform one PPO update step
            stats = ppo_trainer.step(queries, generations, rewards)
            print(f"Query: {query!r} | Reward: {reward:.2f} | Stats: {stats}")

# Execute & Save
if __name__ == "__main__":
    data = load_feedback_data()
    if not data:
        print("No feedback data found. Exiting.")
        exit()

    # Run RLHF for 3 passes over the data
    rlhf_finetune(data, iterations=3)

    # Save the updated model
    save_path = "data/fine_tuned_model_rlhf"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Fine-tuned model saved to {save_path}")
