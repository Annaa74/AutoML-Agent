# automl_flask_app/app.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import time
import uuid
import os
import shutil # For deleting directories

app = Flask(__name__)
# Enable CORS for all routes (for development purposes, be more specific in production)
CORS(app)

# In-memory storage for simulating training jobs
# In a real application, this would be persisted in a database (e.g., SQLite, PostgreSQL)
# {job_id: {"status": "in_progress", "progress": 0, "api_endpoint": null, "error": null, "start_time": timestamp, "prompt": "", "dataset_name": ""}}
training_jobs = {}

# This is where uploaded files and model artifacts would be stored
UPLOAD_FOLDER = 'uploads'
MODEL_ARTIFACTS_FOLDER = 'model_artifacts'

# Create the directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_ARTIFACTS_FOLDER):
    os.makedirs(MODEL_ARTIFACTS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_ARTIFACTS_FOLDER'] = MODEL_ARTIFACTS_FOLDER


@app.route('/')
def index():
    """
    Serves the main frontend HTML page.
    """
    return render_template('index.html')

@app.route('/api/train', methods=['POST']) # Changed to /api/train as per Django views
def train_model_api():
    """
    API endpoint to initiate model training based on natural language prompt and dataset.
    Simulates the process and returns a job ID.
    Handles file upload from FormData.
    """
    try:
        prompt = request.form.get('prompt', '')
        dataset_file = request.files.get('dataset_file')
        
        # Extract advanced options
        model_type_preference = request.form.get('model_type_preference', 'any')
        evaluation_metric = request.form.get('evaluation_metric', 'auto')
        time_limit = request.form.get('time_limit', 60, type=int)
        validation_split = request.form.get('validation_split', 0.2, type=float)
        ensemble_enabled = request.form.get('ensemble_enabled', 'false').lower() == 'true'


        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400
        if not dataset_file:
            return jsonify({"error": "Dataset file is required."}), 400

        # Save the uploaded file temporarily
        original_filename = dataset_file.filename
        # Generate a unique filename to avoid collisions
        stored_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
        dataset_file.save(file_path)
        print(f"File '{original_filename}' saved to {file_path} as {stored_filename}")

        job_id = str(uuid.uuid4())
        training_jobs[job_id] = {
            "status": "initiated",
            "progress": 0,
            "api_endpoint": None,
            "error": None,
            "start_time": time.time(),
            "prompt": prompt,
            "dataset_name": original_filename,
            "stored_filename": stored_filename, # Store the unique filename
            "automl_params": { # Store parsed/advanced params
                "task_type": "classification", # Placeholder, would be parsed by LLM
                "target_variable": "target", # Placeholder, would be parsed by LLM
                "optimization_metric": evaluation_metric,
                "time_limit": time_limit,
                "validation_split": validation_split,
                "ensemble_enabled": ensemble_enabled,
                "key_features_identified": "all columns considered", # Placeholder
                "original_prompt": prompt,
            },
            "top_models_info": [], # For simulated model selection
            "selected_algorithm_name": None,
            "final_metric_value": None,
            "feature_importance_values": {},
            "model_training_time": None,
            "model_artifact_path": None, # Path to the saved model artifact
        }

        # Simulate starting an async task (e.g., using Celery or threading) for actual AutoML
        print(f"Training job {job_id} initiated for prompt: '{prompt}' with dataset: '{original_filename}'")

        return jsonify({"job_id": job_id, "message": "Training process initiated."})

    except Exception as e:
        print(f"Error in /api/train: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<string:job_id>', methods=['GET']) # Changed to /api/status as per Django views
def model_status_api(job_id):
    """
    API endpoint to check the status of a training job.
    Simulates progress and returns API endpoint when complete.
    """
    job = training_jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found."}), 404

    # If already completed or failed, return the stored status
    if job['status'] in ["completed", "failed", "awaiting model selection"]:
        return jsonify(job)

    # Simulate progress based on elapsed time
    elapsed_time = time.time() - job['start_time']
    
    if elapsed_time < 3: # Preprocessing
        job['status'] = "preprocessing data"
        job['progress'] = 20
    elif elapsed_time < 7: # Training initial models
        job['status'] = "training models"
        job['progress'] = 50
    elif elapsed_time < 10: # Awaiting model selection
        job['status'] = "awaiting model selection"
        job['progress'] = 70
        # Simulate top models being ready for selection
        if not job['top_models_info']:
            job['top_models_info'] = [
                {"name": "LightGBM Classifier", "value": "LightGBM", "performance": "0.92 F1-Score", "time": "15s"},
                {"name": "XGBoost Classifier", "value": "XGBoost", "performance": "0.91 F1-Score", "time": "20s"},
                {"name": "Random Forest", "value": "RandomForest", "performance": "0.89 F1-Score", "time": "12s"},
            ]
    elif elapsed_time < 12 and job['status'] == "awaiting model selection":
        # This state is maintained until frontend sends selection.
        # For this simulation, we'll auto-advance if enough time passes.
        # In a real app, you'd have a separate endpoint for selection.
        job['status'] = "finalizing deployment"
        job['progress'] = 85
        job['selected_algorithm_name'] = job['top_models_info'][0]['name'] if job['top_models_info'] else "Auto Selected"
    elif elapsed_time < 15: # Deploying
        job['status'] = "deploying model"
        job['progress'] = 95
    else: # Completed
        job['status'] = "completed"
        job['progress'] = 100
        job['api_endpoint'] = f"https://api.automl-agent.com/predict/{job_id}/"
        job['final_metric_value'] = 0.925 # Simulated final metric
        job['model_training_time'] = round(elapsed_time, 2)
        job['feature_importance_values'] = { # Simulated feature importance
            "feature_A": 0.35, "feature_B": 0.25, "feature_C": 0.15, "feature_D": 0.10, "feature_E": 0.05
        }
        # Simulate saving a model artifact
        model_artifact_filename = f"model_{job_id}.pkl"
        model_artifact_path = os.path.join(app.config['MODEL_ARTIFACTS_FOLDER'], model_artifact_filename)
        # In a real scenario, you'd save the actual model here (e.g., using pickle, joblib)
        with open(model_artifact_path, 'w') as f:
            f.write(f"Simulated model artifact for {job_id}")
        job['model_artifact_path'] = model_artifact_path # Store path for potential deletion
        print(f"Training job {job_id} completed. API: {job['api_endpoint']}")

    return jsonify(job)

@app.route('/api/data_management/upload', methods=['POST'])
def data_management_upload():
    """
    Handles dataset uploads, saves them, and returns stored filename.
    """
    if 'dataset_file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['dataset_file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    if file:
        original_filename = file.filename
        stored_filename = f"{uuid.uuid4()}_{original_filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
        file.save(file_path)
        return jsonify({
            "message": "File uploaded successfully.",
            "original_filename": original_filename,
            "stored_filename": stored_filename
        }), 200
    return jsonify({"error": "File upload failed."}), 500

@app.route('/api/data_management/analyze/<string:stored_filename>', methods=['GET'])
def data_management_analyze(stored_filename):
    """
    Simulates dataset analysis (preview, summary, describe, charts).
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Dataset file not found."}), 404

    # Simulate reading and analyzing the file
    preview_content = "Simulated preview of the dataset...\nColumn1,Column2,Column3\n1,A,X\n2,B,Y\n3,C,Z\n..."
    summary_text = "This is a simulated summary of your dataset. It contains 100 rows and 5 columns. No missing values detected. Key features are Column1 (numerical) and Column2 (categorical)."
    describe_content = "Simulated statistical description:\n             count   mean    std\nColumn1    100.0  50.0  15.0\nColumn3    100.0  0.5   0.2"

    # Simulate chart data (e.g., for a bar chart of categorical distribution or histogram)
    visualization_charts = [
        {
            "type": "bar",
            "data": {
                "labels": ["Category A", "Category B", "Category C"],
                "datasets": [{
                    "label": "Distribution of Category",
                    "data": [30, 50, 20],
                    "backgroundColor": ["#3b82f6", "#22c55e", "#ef4444"]
                }]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"title": {"display": True, "text": "Simulated Categorical Distribution"}}
            }
        },
        {
            "type": "line",
            "data": {
                "labels": ["Jan", "Feb", "Mar", "Apr", "May"],
                "datasets": [{
                    "label": "Simulated Trend",
                    "data": [10, 15, 7, 20, 12],
                    "borderColor": "#f97316",
                    "tension": 0.1
                }]
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"title": {"display": True, "text": "Simulated Numerical Trend"}}
            }
        }
    ]

    return jsonify({
        "preview_html": preview_content,
        "summary_text": summary_text,
        "describe_html": describe_content,
        "visualization_charts": visualization_charts
    }), 200

@app.route('/api/data_management/delete/<string:stored_filename>', methods=['DELETE'])
def data_management_delete(stored_filename):
    """
    Deletes an uploaded dataset file from the Flask backend storage.
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], stored_filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            return jsonify({"message": f"File '{stored_filename}' deleted successfully."}), 200
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File not found."}), 404

@app.route('/api/model_artifacts/delete/<string:artifact_filename>', methods=['DELETE'])
def model_artifacts_delete(artifact_filename):
    """
    Deletes a trained model artifact (or its directory) from the Flask backend storage.
    Assumes artifact_filename could be a directory name if models are saved as folders.
    """
    artifact_path = os.path.join(app.config['MODEL_ARTIFACTS_FOLDER'], artifact_filename)
    if os.path.exists(artifact_path):
        try:
            if os.path.isdir(artifact_path):
                shutil.rmtree(artifact_path) # Delete directory and its contents
                print(f"Deleted model artifact directory: {artifact_path}")
            else:
                os.remove(artifact_path) # Delete file
                print(f"Deleted model artifact file: {artifact_path}")
            return jsonify({"message": f"Model artifact '{artifact_filename}' deleted successfully."}), 200
        except Exception as e:
            print(f"Error deleting model artifact {artifact_path}: {e}")
            return jsonify({"error": f"Failed to delete model artifact: {str(e)}"}), 500
    else:
        return jsonify({"error": "Model artifact not found."}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000) # Ensure Flask runs on port 5000
