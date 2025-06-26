# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import time
from llm_parser import parse_prompt_for_automl
from automl_engine import initiate_automl_training, get_automl_training_status

app = Flask(__name__)
CORS(app) # Enable CORS for all origins, or specify allowed origins

# In-memory store for simulated training jobs (for Sprint 1 & 2)
# In future sprints, this will be replaced by a database (e.g., PostgreSQL)
simulated_training_jobs = {}

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
    
    # Save the uploaded dataset file (temporary for simulation)
    # In a real application, you'd handle secure file storage and larger files.
    filename = secure_filename(dataset_file.filename) # Basic security for filename
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset_file.save(dataset_path)
    print(f"Dataset saved to: {dataset_path}")

    # Step 1: Parse prompt using LLM parser (placeholder)
    try:
        automl_params = parse_prompt_for_automl(prompt)
        print(f"Parsed AutoML parameters: {automl_params}")
    except Exception as e:
        return jsonify({"error": f"Failed to parse prompt: {str(e)}"}), 400

    # Step 2: Initiate AutoML training (simulated)
    job_id = str(uuid.uuid4())
    simulated_training_jobs[job_id] = {
        "status": "initiated",
        "progress": 0,
        "api_endpoint": None, # Will be set upon completion
        "error": None,
        "start_time": time.time(),
        "prompt": prompt,
        "dataset_path": dataset_path, # Store path for simulated engine
        "automl_params": automl_params,
    }

    try:
        # In a real scenario, this would trigger an async task (e.g., Celery)
        initiate_automl_training(job_id, dataset_path, automl_params)
    except Exception as e:
        simulated_training_jobs[job_id]["status"] = "failed"
        simulated_training_jobs[job_id]["error"] = str(e)
        return jsonify({"job_id": job_id, "message": "Training initiation failed.", "error": str(e)}), 500

    return jsonify({"job_id": job_id, "message": "AutoML training initiated successfully."}), 200

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """
    API endpoint to get the status of an AutoML training job.
    """
    job_data = simulated_training_jobs.get(job_id)

    if not job_data:
        return jsonify({"error": "Job not found"}), 404

    # Simulate real-time progress update
    if job_data["status"] not in ["completed", "failed"]:
        elapsed_time = time.time() - job_data["start_time"]
        
        if elapsed_time < 5:
            job_data['status'] = "preprocessing data"
            job_data['progress'] = min(90, int(elapsed_time * 10)) # Progress up to 50%
        elif elapsed_time < 15:
            job_data['status'] = "training models"
            job_data['progress'] = min(90, int(50 + (elapsed_time - 5) * 4)) # Progress from 50% to 90%
        elif elapsed_time < 20:
            job_data['status'] = "deploying model"
            job_data['progress'] = 95
        else:
            job_data['status'] = "completed"
            job_data['progress'] = 100
            # Simulate a deployed API endpoint
            job_data['api_endpoint'] = f"https://api.automl-agent.com/predict/{job_id}"
            print(f"Simulated job {job_id} completed.")
            # Clean up temporary dataset file if needed
            if os.path.exists(job_data['dataset_path']):
                os.remove(job_data['dataset_path'])
                print(f"Cleaned up dataset file: {job_data['dataset_path']}")

    return jsonify(job_data), 200

# Helper function for secure filenames (from Werkzeug)
from werkzeug.utils import secure_filename

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
