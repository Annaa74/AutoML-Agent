# backend/llm_parser.py

def parse_prompt_for_automl(prompt: str) -> dict:
    """
    Parses a natural language prompt to extract AutoML task parameters.
    This is a placeholder for LLM integration (e.g., using Gemini API).

    Args:
        prompt (str): The natural language prompt from the user.

    Returns:
        dict: A dictionary containing extracted task parameters like:
              - 'task_type': e.g., 'classification', 'regression', 'time_series'
              - 'target_variable': The column to predict
              - 'optimization_metric': e.g., 'accuracy', 'f1', 'rmse'
              - 'time_limit': (optional) training time limit in minutes
              - 'validation_split': (optional) ratio for validation data
              - 'other_params': any other specific instructions
    """
    print(f"LLM Parser: Attempting to parse prompt: '{prompt}'")

    # Simple keyword-based parsing for demonstration
    task_type = "classification" if "classify" in prompt.lower() or "category" in prompt.lower() else "regression"
    if "predict sales" in prompt.lower() or "forecast" in prompt.lower():
        task_type = "time_series" # A bit more specific for time series

    target_variable = "target" # Default or placeholder
    if "predict customer churn" in prompt.lower():
        target_variable = "churn"
    elif "forecast quarterly sales" in prompt.lower():
        target_variable = "sales"
    elif "predict house prices" in prompt.lower():
        target_variable = "price"
    
    optimization_metric = "auto"
    if "optimizing for recall" in prompt.lower():
        optimization_metric = "recall"
    elif "optimizing for rmse" in prompt.lower():
        optimization_metric = "rmse"

    time_limit = None
    validation_split = None

    # You could add more sophisticated regex or actual LLM calls here
    # For instance, a Gemini API call:
    # try:
    #     chatHistory = [];
    #     chatHistory.push({ role: "user", parts: [{ text: f"Extract the ML task type, target variable, and optimization metric from this prompt: '{prompt}'. Return as a JSON object with keys 'task_type', 'target_variable', 'optimization_metric'. If not specified, use 'auto'." }] });
    #     const payload = {
    #         contents: chatHistory,
    #         generationConfig: {
    #             responseMimeType: "application/json",
    #             responseSchema: {
    #                 type: "OBJECT",
    #                 properties: {
    #                     "task_type": { "type": "STRING" },
    #                     "target_variable": { "type": "STRING" },
    #                     "optimization_metric": { "type": "STRING" }
    #                 }
    #             }
    #         }
    #     };
    #     const apiKey = "";
    #     const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
    #     const response = await fetch(apiUrl, {
    #                method: 'POST',
    #                headers: { 'Content-Type': 'application/json' },
    #                body: JSON.stringify(payload)
    #            });
    #     const result = await response.json();
    #     if (result.candidates && result.candidates.length > 0 && result.candidates[0].content && result.candidates[0].content.parts && result.candidates[0].content.parts.length > 0) {
    #       const llm_response_json = JSON.parse(result.candidates[0].content.parts[0].text);
    #       task_type = llm_response_json.task_type || task_type;
    #       target_variable = llm_response_json.target_variable || target_variable;
    #       optimization_metric = llm_response_json.optimization_metric || optimization_metric;
    #     }
    # except Exception as e:
    #     print(f"LLM API call failed: {e}")

    return {
        "task_type": task_type,
        "target_variable": target_variable,
        "optimization_metric": optimization_metric,
        "time_limit": time_limit,
        "validation_split": validation_split,
        "original_prompt": prompt,
    }

