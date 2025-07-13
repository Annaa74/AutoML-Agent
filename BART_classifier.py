import torch
from transformers import pipeline, BartForSequenceClassification, BartTokenizer

# Load BART model and tokenizer
model_name = "facebook/bart-large-mnli"
model = BartForSequenceClassification.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Initialize zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

def classify_prompt(prompt, candidate_labels):
    result = classifier(prompt, candidate_labels)
    return result

if __name__ == "__main__":
    # Example usage
    prompt = "I want to build a regression model for predicting house prices."
    candidate_labels = ["regression", "classification", "clustering"]
    classification_result = classify_prompt(prompt, candidate_labels)
    print("Classification Result:", classification_result)
