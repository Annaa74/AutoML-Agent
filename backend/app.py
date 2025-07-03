# backend/app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import time
from werkzeug.utils import secure_filename

from llm_parser import parse_prompt_for_automl
from automl_engine import initiate_automl_training, get_automl_training_status
from data_analyzer import load_dataset_and_get_preview, generate_dataset_summary_llm, generate_visualization_data

app = Flask(__name__)

CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"], supports_credentials=True)

simulated_training_jobs_app_view = {}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return "AutoML Agent Backend is running!"

@app.route('/api/train', methods=['POST'])
def train_model():
    if 'dataset_file' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400
    if 'prompt' not in request.form:
        return jsonify({"error": "No prompt provided"}), 400

    dataset_file = request.files['dataset_file']
    prompt = request.form['prompt']
    
    model_type_preference = request.form.get('model_type_preference')
    evaluation_metric = request.form.get('evaluation_metric')
    time_limit_str = request.form.get('time_limit')
    validation_split_str = request.form.get('validation_split')
    ensemble_enabled_str = request.form.get('ensemble_enabled', 'false')

    time_limit = int(time_limit_str) if time_limit_str else None
    validation_split = float(validation_split_str) if validation_split_str else None
    ensemble_enabled = ensemble_enabled_str.lower() == 'true'

    filename = secure_filename(dataset_file.filename)
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dataset_file.save(dataset_path)
    print(f"Dataset saved to: {dataset_path}")

    try:
        automl_params = parse_prompt_for_automl(
            prompt,
            time_limit=time_limit,
            validation_split=validation_split,
            ensemble_enabled=ensemble_enabled
        )
        if model_type_preference and model_type_preference != 'any':
            automl_params['task_type'] = model_type_preference
        if evaluation_metric and evaluation_metric != 'auto':
            automl_params['optimization_metric'] = evaluation_metric
        
        print(f"Parsed AutoML parameters (final): {automl_params}")
    except Exception as e:
        return jsonify({"error": f"Failed to parse prompt: {str(e)}"}), 400

    job_id = str(uuid.uuid4())
    simulated_training_jobs_app_view[job_id] = {
        "status": "initiated",
        "progress": 0,
        "api_endpoint": None,
        "error": None,
        "start_time": time.time(),
        "prompt": prompt,
        "dataset_path": dataset_path,
        "automl_params": automl_params,
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
    job_data = get_automl_training_status(job_id)
    if not job_data:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job_data), 200


# --- Data Management Endpoints ---

@app.route('/api/data_management/upload', methods=['POST'])
def upload_data_for_analysis():
    if 'dataset_file' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400

    dataset_file = request.files['dataset_file']
    if dataset_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    original_filename = secure_filename(dataset_file.filename)
    unique_filename = f"{uuid.uuid4()}_{original_filename}" # Use original filename as part of unique filename
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    dataset_file.save(dataset_path)
    print(f"Dataset for analysis saved to: {dataset_path}")

    # Return both original and unique filename
    return jsonify({
        "message": "Dataset uploaded successfully for analysis.",
        "original_filename": original_filename,
        "stored_filename": unique_filename # Use "stored_filename" to be explicit for Django
    }), 200

@app.route('/api/data_management/analyze/<filename>', methods=['GET'])
def analyze_dataset(filename):
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset not found."}), 404

    try:
        df, head_html, info_str, describe_html = load_dataset_and_get_preview(dataset_path)
        summary_text = generate_dataset_summary_llm(df, info_str, describe_html)
        visualization_data = generate_visualization_data(df)

        return jsonify({
            "message": "Analysis complete.",
            "filename": filename,
            "preview_html": head_html,
            "info_text": info_str,
            "describe_html": describe_html,
            "summary_text": summary_text,
            "visualization_charts": visualization_data
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
