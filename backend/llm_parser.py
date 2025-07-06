# backend/llm_parser.py

import json # Needed if BART output is JSON
# Import necessary libraries for BART
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import torch

# Determine device for BART inference
# device = 0 if torch.cuda.is_available() else -1 # 0 for GPU, -1 for CPU. Use -1 for CPU only.
# For simplicity and wider compatibility, we'll assume CPU usage or let pipeline handle it.
# If you have a GPU and want to use it, uncomment and ensure torch is installed with CUDA support.

# Initialize BART Zero-Shot Classifier (loaded once when module is imported)
# This model is good for classification without explicit training data.
classifier = None # Initialize to None
try:
    # Using CPU for broader compatibility in Canvas environment
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("BART 'facebook/bart-large-mnli' zero-shot classifier loaded successfully.")
except ImportError:
    print("Transformers or PyTorch not installed correctly. BART Zero-Shot Classifier will not be available.")
except Exception as e:
    print(f"Error loading BART Zero-Shot Classifier: {e}. It will not be available.")

# You could also load a text generation/summarization model for more complex parsing
# model_name_for_generation = "facebook/bart-large-cnn"
# generator = None # Initialize to None
# try:
#     tokenizer_gen = BartTokenizer.from_pretrained(model_name_for_generation)
#     model_gen = BartForConditionalGeneration.from_pretrained(model_name_for_generation)
#     generator = pipeline("text2text-generation", model=model_gen, tokenizer=tokenizer_gen)
#     print(f"BART '{model_name_for_generation}' generator loaded successfully.")
# except ImportError:
#     print("Transformers or PyTorch not installed correctly. BART generator will not be available.")
# except Exception as e:
#     print(f"Error loading BART generator: {e}. It will not be available.")


def parse_prompt_for_automl(prompt: str, time_limit=None, validation_split=None, ensemble_enabled=False) -> dict:
    """
    Parses a natural language prompt to extract AutoML task parameters using BART (conceptually).

    Args:
        prompt (str): The natural language prompt from the user.
        time_limit (int, optional): User-specified time limit in minutes.
        validation_split (float, optional): User-specified validation split ratio.
        ensemble_enabled (bool, optional): User-specified ensembling preference.

    Returns:
        dict: A dictionary containing extracted task parameters.
    """
    print(f"LLM Parser: Attempting to parse prompt: '{prompt}' using BART (conceptually).")

    # Initialize with default/fallback values
    task_type = "classification"
    target_variable = "target"
    optimization_metric = "auto"
    key_features = "general features, all columns considered"

    # --- Use BART for Task Type Classification (if loaded successfully) ---
    if classifier:
        candidate_labels = ["classification", "regression", "time series"]
        try:
            # Perform zero-shot classification
            result = classifier(prompt, candidate_labels)
            # Find the label with the highest score
            if result['scores'][0] > 0.6: # Confidence threshold
                detected_task_type = result['labels'][0]
                # Map BART's output to your internal task types if necessary
                if detected_task_type == "time series":
                    task_type = "time_series"
                else:
                    task_type = detected_task_type
                print(f"BART detected task type: {task_type} (Confidence: {result['scores'][0]:.2f})")
            else:
                print(f"BART confidence too low for task type detection. Falling back to keyword parsing.")
        except Exception as e:
            print(f"Error during BART zero-shot classification: {e}. Falling back to keyword parsing.")
    
    # --- Keyword-based / Rule-based parsing for other parameters (fallback or if BART isn't fine-tuned for this) ---
    # This part remains mostly keyword-based because getting precise target variable names
    # and specific feature lists directly from a general BART model in JSON without
    # fine-tuning or complex prompt engineering is difficult.

    # Refine task_type based on explicit keywords if not confident from BART
    if "predict customer churn" in prompt.lower() or "churn prediction" in prompt.lower() or "customer retention" in prompt.lower():
        task_type = "classification"
        target_variable = "Churn" # Assuming 'Churn' is a common column name in churn datasets
        optimization_metric = "recall"
        key_features = "gender, tenure, monthly_charges, total_charges, contract_type, internet_service"
    elif "forecast sales" in prompt.lower() or "predict sales" in prompt.lower() or "sales forecasting" in prompt.lower():
        task_type = "time_series"
        target_variable = "Sales"
        optimization_metric = "mae"
        key_features = "date, promotional_spend, economic_indicators, seasonality"
    elif "predict house prices" in prompt.lower() or "housing price" in prompt.lower():
        task_type = "regression"
        target_variable = "Price"
        optimization_metric = "rmse"
        key_features = "location, square_footage, number_of_bedrooms, year_built, property_type"
    elif "classify reviews" in prompt.lower() or "sentiment" in prompt.lower():
        task_type = "classification"
        target_variable = "Sentiment"
        optimization_metric = "f1_score"
        key_features = "text_length, keyword_frequency, sentiment_score"

    # For optimization metric, if not detected by BART or specific keywords
    if "optimize for accuracy" in prompt.lower():
        optimization_metric = "accuracy"
    elif "optimize for f1" in prompt.lower() or "f1-score" in prompt.lower():
        optimization_metric = "f1_score"
    elif "optimize for rmse" in prompt.lower():
        optimization_metric = "rmse"
    elif "optimize for mae" in prompt.lower():
        optimization_metric = "mae"
    
    # --- Advanced LLM Extraction (Requires fine-tuning or very clever prompt engineering for BART) ---
    # If you had a BART model fine-tuned for JSON extraction:
    # if generator:
    #     try:
    #         llm_prompt_for_json = f"""
    #         Extract the following from the user prompt:
    #         - ML task type (classification, regression, time_series)
    #         - Target variable name
    #         - Optimization metric (accuracy, f1_score, rmse, mae, or auto)
    #         - Key features for analysis (comma-separated list, infer if not explicit)
    #         User prompt: "{prompt}"
    #         Output in JSON format. Example: {{"task_type": "classification", "target_variable": "Churn", "optimization_metric": "recall", "key_features_identified": "tenure, contract_type"}}
    #         """
    #         generated_response = generator(llm_prompt_for_json, max_length=200, num_beams=5, early_stopping=True)
    #         generated_text = generated_response[0]['generated_text']
    #         
    #         # Attempt to parse the generated JSON
    #         parsed_data = json.loads(generated_text)
    #         task_type = parsed_data.get('task_type', task_type)
    #         target_variable = parsed_data.get('target_variable', target_variable)
    #         optimization_metric = parsed_data.get('optimization_metric', optimization_metric)
    #         key_features = parsed_data.get('key_features_identified', key_features)
    #         print(f"BART (generation) parsed: {parsed_data}")
    #     except json.JSONDecodeError:
    #         print("BART generated invalid JSON. Falling back to keyword parsing.")
    #     except Exception as e:
    #         print(f"Error during BART generation/parsing: {e}. Falling back to keyword parsing.")


    return {
        "task_type": task_type,
        "target_variable": target_variable,
        "optimization_metric": optimization_metric,
        "time_limit": time_limit, # These are passed from frontend advanced options
        "validation_split": validation_split, # These are passed from frontend advanced options
        "original_prompt": prompt,
        "key_features_identified": key_features,
        "ensemble_enabled": ensemble_enabled, # Passed from frontend
    }

