# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import time
from werkzeug.utils import secure_filename # For secure filenames

# Import the updated llm_parser and automl_engine
from llm_parser import parse_prompt_for_automl
from automl_engine import initiate_automl_training, get_automl_training_status

app = Flask(__name__)

# Configure CORS explicitly for development.
# In production, specify exact origins, not '*'.
CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"], supports_credentials=True)

# In-memory store for tracking simulated training jobs.
# The actual detailed job data will reside in automl_engine's _active_automl_jobs
# This remains for consistency in Flask app's view.
simulated_training_jobs_app_view = {}

# Placeholder for uploaded datasets. In a real app, use persistent storage.
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    """Basic endpoint to check if Flask app is running."""
    return "AutoML Agent Backend is running!"

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    API endpoint to initiate AutoML model training.
    Expects a natural language prompt and a dataset file.
    """
    if 'dataset_file' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400
    if 'prompt' not in request.form:
        return jsonify({"error": "No prompt provided"}), 400

    dataset_file = request.files['dataset_file']
    prompt = request.form['prompt']
    
    # Get advanced options from form data, default to None/False if not provided
    model_type_preference = request.form.get('model_type_preference')
    evaluation_metric = request.form.get('evaluation_metric')
    time_limit_str = request.form.get('time_limit')
    validation_split_str = request.form.get('validation_split')
    ensemble_enabled_str = request.form.get('ensemble_enabled', 'false') # Default to 'false'

    # Convert types
    time_limit = int(time_limit_str) if time_limit_str else None
    validation_split = float(validation_split_str) if validation_split_str else None
    ensemble_enabled = ensemble_enabled_str.lower() == 'true'

    # Save the uploaded dataset file
    filename = secure_filename(dataset_file.filename)
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset_file.save(dataset_path)
    print(f"Dataset saved to: {dataset_path}")

    # Step 1: Parse prompt using LLM parser (or simulate parsing)
    try:
        # Pass advanced options from frontend directly to parser for inclusion in automl_params
        automl_params = parse_prompt_for_automl(
            prompt,
            time_limit=time_limit,
            validation_split=validation_split,
            ensemble_enabled=ensemble_enabled
        )
        # Override LLM-parsed defaults if explicit advanced options were provided
        if model_type_preference and model_type_preference != 'any':
            automl_params['task_type'] = model_type_preference
        if evaluation_metric and evaluation_metric != 'auto':
            automl_params['optimization_metric'] = evaluation_metric
        # time_limit and validation_split are already passed from form
        
        print(f"Parsed AutoML parameters (final): {automl_params}")
    except Exception as e:
        return jsonify({"error": f"Failed to parse prompt: {str(e)}"}), 400

    # Step 2: Initiate AutoML training
    job_id = str(uuid.uuid4())
    # Update app's internal view of job status (automl_engine will hold full details)
    simulated_training_jobs_app_view[job_id] = {
        "status": "initiated",
        "progress": 0,
        "api_endpoint": None,
        "error": None,
        "start_time": time.time(),
        "prompt": prompt,
        "dataset_path": dataset_path,
        "automl_params": automl_params, # Store params passed to engine
    }

    try:
        initiate_automl_training(job_id, dataset_path, automl_params)
    except Exception as e:
        simulated_training_jobs_app_view[job_id]["status"] = "failed"
        simulated_training_jobs_app_view[job_id]["error"] = str(e)
        return jsonify({"job_id": job_id, "message": "Training initiation failed.", "error": str(e)}), 500

    return jsonify({"job_id": job_id, "message": "AutoML training initiated successfully."}), 200

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    API endpoint to get the status of an AutoML training job.
    Retrieves the actual status from the automl_engine's internal store.
    """
    job_data = get_automl_training_status(job_id)

    if not job_data:
        return jsonify({"error": "Job not found"}), 404

    # The automl_engine is now responsible for updating its own status/progress
    # and will provide all the necessary details.
    return jsonify(job_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
