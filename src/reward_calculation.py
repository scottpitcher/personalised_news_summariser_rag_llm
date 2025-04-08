# reward_calculation.py

def calculate_reward(feedback):
    """
    Converts user feedback into a numerical reward.

    Feedback mapping:
      - "thumbs_up" returns +1.0 (indicating a positive reaction)
      - "thumbs_down" returns -1.0 (indicating a negative reaction)
      - A rewrite suggestion (provided as a dictionary with key "rewrite") returns -0.5 
        (the idea is that if a rewrite is suggested, the original output should be penalized,
         but you may adjust this value based on your experiments)
    
    Any unrecognized feedback returns 0.0.
    """
    if isinstance(feedback, str):
        if feedback == "thumbs_up":
            return 1.0
        elif feedback == "thumbs_down":
            return -1.0
    elif isinstance(feedback, dict) and "rewrite" in feedback:
        return -0.5  # Adjust this penalty if needed.
    return 0.0

# Example tests:
if __name__ == "__main__":
    print("Feedback 'thumbs_up':", calculate_reward("thumbs_up"))             # Expected output: 1.0
    print("Feedback 'thumbs_down':", calculate_reward("thumbs_down"))         # Expected output: -1.0
    print("Feedback {rewrite: 'Needs a friendlier tone'}:", calculate_reward({"rewrite": "Needs a friendlier tone"}))  # Expected output: -0.5
