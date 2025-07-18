# automl_flask_app/app.py
from flask import Flask, render_template, request, jsonify, after_this_request, send_file
from flask_cors import CORS # For handling Cross-Origin Resource Sharing
import time
import uuid
import os
import shutil # For deleting directories

app = Flask(__name__)

# --- IMPORTANT CORS CONFIGURATION ---
# Specify the exact origin(s) of your frontend application(s).
# If your Django frontend runs on http://127.0.0.1:8000, use that.
# If it runs on localhost, use http://localhost:8000.
# For multiple origins, use a list: origins=["http://127.0.0.1:8000", "http://localhost:8000"]
# Also, set supports_credentials=True to allow cookies (like CSRF token) to be sent.
CORS(app, origins=["http://127.0.0.1:8000", "http://localhost:8000"], supports_credentials=True)

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

# --- Debugging: Print all response headers and explicitly set Access-Control-Allow-Credentials ---
@app.after_request
def after_request_func(response):
    # Ensure Access-Control-Allow-Origin is set to the requesting origin if credentials are included
    origin = request.headers.get('Origin')
    if origin and origin in ["http://127.0.0.1:8000", "http://localhost:8000"]:
        response.headers.add('Access-Control-Allow-Origin', origin)
    
    # Explicitly set Access-Control-Allow-Credentials to 'true'
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    
    # Allow specific headers (especially Content-Type and X-CSRFToken for preflight)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,X-CSRFToken')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS') # Ensure OPTIONS is allowed

    print(f"\n--- Response Headers for {request.path} ---")
    for header, value in response.headers.items():
        print(f"{header}: {value}")
    print("---------------------------------------\n")
    return response


@app.route('/')
def index():
    """
    Returns a simple message indicating the Flask API is running.
    The main frontend HTML page (intro_page.html) is served by Django.
    """
    return jsonify({"message": "AutoML Agent Flask API is running!"})

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
        # Simulate creating a dummy model file
        model_artifact_filename = f"model_{job_id}.pkl"
        model_artifact_path = os.path.join(app.config['MODEL_ARTIFACTS_FOLDER'], model_artifact_filename)
        with open(model_artifact_path, 'w') as f:
            f.write(f"Simulated model artifact content for {job_id}") # Write some dummy content
        print(f"Simulated model artifact created at: {model_artifact_path}")


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
            "model_artifact_path": model_artifact_path, # Store path to the dummy model artifact
        }

        # Simulate starting an async task (e.g., using Celery or threading) for actual AutoML
        print(f"Training job {job_id} initiated for prompt: '{prompt}' with dataset: '{original_filename}'")

        return jsonify({"job_id": job_id, "message": "Training process initiated.", "stored_filename": stored_filename, "original_filename": original_filename})

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
        # model_artifact_path is already set when job is initiated
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
    describe_content = "Simulated statistical description:\n             count   mean    std\nColumn1    100.0   50.0   15.0\nColumn3    100.0   0.5    0.2"

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

@app.route('/api/download_model/<string:job_id>', methods=['GET'])
def download_model(job_id):
    """
    Serves the trained model artifact (.pkl file) for download.
    """
    job = training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Training job not found."}), 404

    model_path = job.get('model_artifact_path')
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "Model artifact not found for this job."}), 404

    try:
        # Extract the filename from the path to use as the download name
        filename = os.path.basename(model_path)
        return send_file(model_path, as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"Error serving model file for job {job_id}: {e}")
        return jsonify({"error": f"Failed to download model: {str(e)}"}), 500


# Simulated authentication endpoints
@app.route('/api/login/', methods=['POST'])
def login_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Basic simulation: check for a hardcoded user or any non-empty credentials
    if username == "user" and password == "password":
        # In a real app, you'd verify credentials against a database
        # and then set up a session or issue a token (e.g., JWT)
        return jsonify({"message": "Login successful!", "redirect_url": "/dashboard"}), 200
    elif username and password:
        # Simulate successful login for any non-empty credentials for demo purposes
        return jsonify({"message": "Login successful!", "redirect_url": "/dashboard"}), 200
    else:
        return jsonify({"message": "Invalid username or password."}), 401

@app.route('/api/signup/', methods=['POST'])
def signup_api():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Username and password are required."}), 400
    
    # Basic simulation: always "succeed" for demo purposes, or add basic checks
    # In a real app, you'd save the new user to a database, hash the password, etc.
    if len(password) < 6:
        return jsonify({"message": "Password must be at least 6 characters long."}), 400
    
    # Simulate user already exists
    if username == "existing_user": # Example of a simulated existing user
        return jsonify({"message": "Username already taken."}), 409 # Conflict
    
    return jsonify({"message": "Signup successful! Please log in."}), 201 # Created

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Ensure Flask runs on port 5000
