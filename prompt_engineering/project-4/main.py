import random
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def generate_outputs(query):
    responses = [
        "Flu symptoms include fever, cough, sore throat, and fatigue.",
        "Common flu symptoms are high fever, muscle aches, and a runny nose.",
        "The flu can cause symptoms such as chills, headache, and loss of appetite."
    ]
    return responses

def collect_feedback(responses):
    feedback = {
        "Flu symptoms include fever, cough, sore throat, and fatigue.": {"clarity": 5, "accuracy": 5, "tone": 4},
        "Common flu symptoms are high fever, muscle aches, and a runny nose.": {"clarity": 5, "accuracy": 5, "tone": 4},
        "The flu can cause symptoms such as chills, headache, and loss of appetite.": {"clarity": 4, "accuracy": 5, "tone": 3}
    }
    return feedback

def train_reward_model(feedback):
    reward_scores = {response: sum(scores.values()) / len(scores) for response, scores in feedback.items()}
    return reward_scores

def static_prompt_with_dynamic_input(query):
    static_prompt = "You are a healthcare chatbot. Provide a clear and accurate response to the user's query about flu symptoms."
    dynamic_input = f"User Query: '{query}'"
    enhanced_prompt = f"{static_prompt} Ensure the response is empathetic and lists common symptoms. {dynamic_input}"
    return enhanced_prompt

def chain_of_thought_prompt(query):
    cot_prompt = (
        "Think step-by-step:\n"
        "1. Identify the symptoms mentioned: fever and sore throat.\n"
        "2. Recall common flu symptoms.\n"
        "3. Determine if the mentioned symptoms match common flu symptoms.\n"
        "4. Provide a reassuring response based on the match."
    )
    return cot_prompt

def detect_bias(response):
    biased_phrases = ["you should", "always", "never"]
    bias_detected = any(phrase in response.lower() for phrase in biased_phrases)
    return bias_detected

def anonymize_data(query):
    anonymized_query = query.replace("my", "the patient's")
    return anonymized_query

def evaluate_metrics(feedback):
    clarity = sum(scores["clarity"] for scores in feedback.values()) / len(feedback)
    accuracy = sum(scores["accuracy"] for scores in feedback.values()) / len(feedback)
    tone = sum(scores["tone"] for scores in feedback.values()) / len(feedback)
    return {"clarity": clarity, "accuracy": accuracy, "tone": tone}

query = "What are the symptoms of the flu?"
responses = generate_outputs(query)
feedback = collect_feedback(responses)
reward_scores = train_reward_model(feedback)

metrics = evaluate_metrics(feedback)
print("Evaluation Metrics:", metrics)

for response in responses:
    if detect_bias(response):
        print(f"Bias detected in response: {response}")

anonymized_query = anonymize_data(query)
print("Anonymized Query:", anonymized_query)

enhanced_prompt = static_prompt_with_dynamic_input(query)
cot_prompt = chain_of_thought_prompt(query)
print("Enhanced Prompt:", enhanced_prompt)
print("Chain-of-Thought Prompt:", cot_prompt)
