# backend/llm_parser.py

# Uncomment and install 'google-generativeai' if you want to use a live Gemini API
import google.generativeai as genai

def parse_prompt_for_automl(prompt: str, time_limit=None, validation_split=None, ensemble_enabled=False) -> dict:
    """
    Parses a natural language prompt to extract AutoML task parameters.
    This version includes placeholders for where a real LLM integration would go.

    Args:
        prompt (str): The natural language prompt from the user.
        time_limit (int, optional): User-specified time limit in minutes.
        validation_split (float, optional): User-specified validation split ratio.
        ensemble_enabled (bool, optional): User-specified ensembling preference.

    Returns:
        dict: A dictionary containing extracted task parameters like:
              - 'task_type': e.g., 'classification', 'regression', 'time_series'
              - 'target_variable': The column to predict
              - 'optimization_metric': e.g., 'accuracy', 'f1', 'rmse'
              - 'time_limit': (optional) training time limit in minutes
              - 'validation_split': (optional) ratio for validation data
              - 'key_features_identified': (simulated) LLM's interpretation of important features.
              - 'ensemble_enabled': (bool) whether ensembling is requested
    """
    print(f"LLM Parser: Attempting to parse prompt: '{prompt}'")

    # --- START: Simulated LLM Parsing Logic ---
    # # In a real application, this entire block would be replaced by an LLM API call.
    # # The LLM would analyze the 'prompt' and return structured JSON.

    # # Default values
    # task_type = "classification" # Common default for structured data
    # target_variable = "target"
    # optimization_metric = "auto"
    # key_features = "general features, all columns considered"

    # # Simple keyword-based logic for demonstration purposes
    # if "predict customer churn" in prompt.lower() or "churn prediction" in prompt.lower() or "customer retention" in prompt.lower():
    #     task_type = "classification"
    #     target_variable = "Churn" # Assuming 'Churn' is the column name in dataset
    #     optimization_metric = "recall" # Good for imbalanced classification
    #     key_features = "gender, tenure, monthly_charges, total_charges, contract_type, internet_service"
    # elif "forecast sales" in prompt.lower() or "predict sales" in prompt.lower() or "sales forecasting" in prompt.lower():
    #     task_type = "time_series"
    #     target_variable = "Sales" # Assuming 'Sales' is the column name
    #     optimization_metric = "mae"
    #     key_features = "date, promotional_spend, economic_indicators, seasonality"
    # elif "predict house prices" in prompt.lower() or "housing price" in prompt.lower():
    #     task_type = "regression"
    #     target_variable = "Price" # Assuming 'Price' is the column name
    #     optimization_metric = "rmse"
    #     key_features = "location, square_footage, number_of_bedrooms, year_built, property_type"
    # elif "classify reviews" in prompt.lower() or "sentiment" in prompt.lower():
    #     task_type = "classification"
    #     target_variable = "Sentiment"
    #     optimization_metric = "f1_score"
    #     key_features = "text_length, keyword_frequency, sentiment_score"

    # --- END: Simulated LLM Parsing Logic ---

    # --- START: Example of Actual LLM API Call (Commented Out) ---
    # This is how you would replace the simulated parsing above with a real LLM call.
    # Ensure 'google-generativeai' is installed and you have a valid API key.

    try:
        # Configure your Gemini API key (replace with your actual key or environment variable)
        # genai.configure(api_key="YOUR_GEMINI_API_KEY") # Or set as GOOGLE_API_KEY env var
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define the JSON schema for the expected output
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "task_type": { "type": "STRING", "description": "Classification, regression, or time_series" },
                "target_variable": { "type": "STRING", "description": "The name of the column to predict" },
                "optimization_metric": { "type": "STRING", "description": "e.g., accuracy, f1_score, rmse, mae. Use 'auto' if not specified." },
                "key_features_identified": { "type": "STRING", "description": "Comma-separated list of most important features from the prompt, if identifiable." }
            },
            "required": ["task_type", "target_variable", "optimization_metric"]
        }
        
        # Construct the prompt for the LLM
        llm_prompt = f"""
        Analyze the following user request for an AutoML task and extract the key parameters.
        User Prompt: "{prompt}"
        
        Provide the output as a JSON object with the following keys:
        'task_type': (e.g., 'classification', 'regression', 'time_series')
        'target_variable': (The name of the column to predict, infer if not explicit)
        'optimization_metric': (e.g., 'accuracy', 'f1_score', 'rmse', 'mae'. Use 'auto' if not specified.)
        'key_features_identified': (A comma-separated list of the most important features mentioned or implied for analysis, if identifiable. If not, use 'N/A'.)
        
        Ensure the output strictly adheres to the JSON schema provided.
        """
        
        response = model.generate_content(
            llm_prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }
        )
        
        # Parse the LLM's JSON response
        llm_response_json = json.loads(response.text)
        
        task_type = llm_response_json.get('task_type', task_type)
        target_variable = llm_response_json.get('target_variable', target_variable)
        optimization_metric = llm_response_json.get('optimization_metric', optimization_metric)
        key_features = llm_response_json.get('key_features_identified', key_features)
        
    except Exception as e:
        print(f"LLM API call failed or parsing error: {e}")
        # Fallback to simulated/default values if API call fails
        pass
    # --- END: Example of Actual LLM API Call (Commented Out) ---

    return {
        "task_type": task_type,
        "target_variable": target_variable,
        "optimization_metric": optimization_metric,
        "time_limit": time_limit, # Passed from frontend, not parsed by LLM here
        "validation_split": validation_split, # Passed from frontend, not parsed by LLM here
        "original_prompt": prompt,
        "key_features_identified": key_features,
        "ensemble_enabled": ensemble_enabled, # Passed from frontend
    }

