# backend/automl_engine.py

import pandas as pd
import time
import os
import random
import uuid # For unique model filenames
import json
import threading # To run AutoML in a separate thread

# PyCaret imports - dynamically import based on task type
from pycaret.classification import setup as classification_setup
from pycaret.classification import compare_models as classification_compare_models
from pycaret.classification import create_model as classification_create_model
from pycaret.classification import tune_model as classification_tune_model
from pycaret.classification import blend_models as classification_blend_models
from pycaret.classification import finalize_model as classification_finalize_model
from pycaret.classification import save_model as classification_save_model
from pycaret.classification import pull as classification_pull # For metrics

from pycaret.regression import setup as regression_setup
from pycaret.regression import compare_models as regression_compare_models
from pycaret.regression import create_model as regression_create_model
from pycaret.regression import tune_model as regression_tune_model
from pycaret.regression import blend_models as regression_blend_models
from pycaret.regression import finalize_model as regression_finalize_model
from pycaret.regression import save_model as regression_save_model
from pycaret.regression import pull as regression_pull # For metrics

# PyCaret for time series is experimental, so using a placeholder setup
# from pycaret.time_series import setup as time_series_setup
# from pycaret.time_series import compare_models as time_series_compare_models
# from pycaret.time_series import finalize_model as time_series_finalize_model
# from pycaret.time_series import save_model as time_series_save_model


# Directory to save trained models
MODELS_DIR = 'trained_models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# In-memory store for real-time job status (would be database in production)
_active_automl_jobs = {}

def _get_pycaret_module(task_type: str):
    """Helper to get the correct PyCaret module based on task type."""
    if task_type == 'classification':
        return {
            'setup': classification_setup,
            'compare_models': classification_compare_models,
            'create_model': classification_create_model,
            'tune_model': classification_tune_model,
            'blend_models': classification_blend_models,
            'finalize_model': classification_finalize_model,
            'save_model': classification_save_model,
            'pull': classification_pull,
            'metric_name': 'Accuracy' # Default metric name for display
        }
    elif task_type == 'regression':
        return {
            'setup': regression_setup,
            'compare_models': regression_compare_models,
            'create_model': regression_create_model,
            'tune_model': regression_tune_model,
            'blend_models': regression_blend_models,
            'finalize_model': regression_finalize_model,
            'save_model': regression_save_model,
            'pull': regression_pull,
            'metric_name': 'RMSE' # Default metric name for display
        }
    elif task_type == 'time_series':
        # PyCaret time_series is more complex and has different API,
        # For simplicity in this demo, we'll simulate it similar to classification/regression
        # In a real app, this would be a distinct PyCaret time_series workflow.
        print("Warning: PyCaret time series integration is a placeholder simulation for now.")
        return {
            'setup': classification_setup, # Using classification setup as placeholder
            'compare_models': classification_compare_models, # Placeholder
            'create_model': classification_create_model, # Placeholder
            'tune_model': classification_tune_model, # Placeholder
            'blend_models': classification_blend_models, # Placeholder
            'finalize_model': classification_finalize_model, # Placeholder
            'save_model': classification_save_model, # Placeholder
            'pull': classification_pull, # Placeholder
            'metric_name': 'MAE' # Common metric for time series
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def _run_automl_pipeline(job_id: str, dataset_path: str, params: dict):
    """
    Executes the actual PyCaret AutoML pipeline in a separate thread/process
    (simulated as a direct call for simplicity in this single-file context).
    """
    job_data = _active_automl_jobs[job_id]
    
    try:
        df = pd.read_csv(dataset_path) # Assuming CSV for simplicity
        job_data['status'] = "preprocessing data (PyCaret setup)"
        job_data['progress'] = 10
        time.sleep(2) # Simulate work

        pycaret_module = _get_pycaret_module(params['task_type'])
        
        # PyCaret Setup
        # Handle potential errors during setup (e.g., target not found, data issues)
        try:
            exp = pycaret_module['setup'](
                data=df,
                target=params['target_variable'],
                session_id=random.randint(1, 10000), # Reproducibility
                # Removed 'silent=True' and 'train_size' as they are incompatible with PyCaret 3.x
                # PyCaret will handle internal data splitting for cross-validation by default
                n_jobs=-1 # Use all available cores
            )
        except Exception as setup_e:
            raise ValueError(f"PyCaret setup failed: {setup_e}. Check target variable and data types.")

        job_data['status'] = "comparing models"
        job_data['progress'] = 30
        time.sleep(3) # Simulate work

        # Compare Models
        best_models_list = pycaret_module['compare_models'](
            n_select=3, # Get top 3 models
            sort=params['optimization_metric'] if params['optimization_metric'] != 'auto' else pycaret_module['metric_name'],
            # For time series, there might be different models/metrics
            turbo=True, # Faster comparison
            errors='ignore' # Continue even if some models fail
        )
        
        if not best_models_list:
            raise ValueError("No models could be compared or found by PyCaret. Check dataset and task.")

        # Extract top models info for user selection
        top_models_info = []
        # PyCaret's compare_models returns a list of estimators.
        # Use pycaret.pull() to get the metrics from the last executed comparison.
        compare_df = pycaret_module['pull']() # Get the results dataframe

        if compare_df.empty:
            raise ValueError("PyCaret compare_models did not return any results. Check dataset/task.")

        # Ensure that compare_df has the expected columns for sorting and displaying
        required_cols = [pycaret_module['metric_name'], 'Time (min)'] # Time might not be directly available for all models
        if pycaret_module['metric_name'] not in compare_df.columns:
            # Fallback if primary metric column is missing (e.g., for some specific PyCaret versions/tasks)
            print(f"Warning: Metric '{pycaret_module['metric_name']}' not found in PyCaret comparison results. Using simulated performance.")

        for i, model_estimator in enumerate(best_models_list):
            model_name = type(model_estimator).__name__
            
            # Try to get actual performance from compare_df
            actual_performance = "N/A"
            if model_name in compare_df.index and pycaret_module['metric_name'] in compare_df.columns:
                actual_performance = round(compare_df.loc[model_name][pycaret_module['metric_name']], 4)
            else:
                actual_performance = round(random.uniform(0.75, 0.95), 4) # Fallback simulated

            # PyCaret compare_models usually doesn't provide individual training time directly in the output dataframe.
            # So, we simulate training time for display.
            simulated_time = random.randint(5, 25) # Simulate time

            top_models_info.append({
                'name': model_name,
                'performance': f"{pycaret_module['metric_name']}: {actual_performance}",
                'time': f"{simulated_time} min",
                'value': model_name # Use model name as value for selection
            })
            if i >= 2: break # Limit to top 3 for display

        job_data['top_models_info'] = top_models_info
        job_data['status'] = "awaiting model selection"
        job_data['progress'] = 70 # Pause for user selection
        print(f"AutoML Engine: Job {job_id} reached model selection. Top models: {top_models_info}")
        
        # At this point, the frontend will pause and wait for user input.
        # The backend simulation here continues as if a selection will be made.
        # In a real async system, the backend would wait for a "selection" API call.
        
        time.sleep(5) # Simulate user thinking time on frontend

        # Finalize Model based on what would be the best or an ensemble
        job_data['status'] = "finalizing model & deployment"
        job_data['progress'] = 80
        time.sleep(3) # Simulate work
        
        final_model = None
        selected_algo_for_finalization = best_models_list[0] # Default to the top model
        job_data['selected_algorithm_name'] = type(selected_algo_for_finalization).__name__

        # If ensembling was enabled from frontend, simulate blending
        if params['ensemble_enabled'] and params['task_type'] in ['classification', 'regression'] and len(best_models_list) >= 2:
            print(f"AutoML Engine: Attempting to blend models for job {job_id}")
            # Create individual best models first to blend them
            # For simplicity, blend a couple of the best ones
            models_to_blend = [pycaret_module['create_model'](algo) for algo in best_models_list[:3]] # Blend top 3
            final_model = pycaret_module['blend_models'](models_to_blend, optimize=pycaret_module['metric_name'])
            job_data['selected_algorithm_name'] = "Blended Ensemble Model"
        else:
            final_model = pycaret_module['finalize_model'](selected_algo_for_finalization)
            job_data['selected_algorithm_name'] = type(selected_algo_for_finalization).__name__

        job_data['status'] = "saving model"
        job_data['progress'] = 90
        time.sleep(2) # Simulate work

        # Save the model
        model_filename = os.path.join(MODELS_DIR, f"model_{job_id}.pkl")
        pycaret_module['save_model'](final_model, model_filename, verbose=False)
        job_data['model_artifact_path'] = model_filename
        print(f"AutoML Engine: Model for job {job_id} saved to {model_filename}")

        # Simulate feature importance values.
        # In PyCaret, you'd typically get feature importance after finalize_model,
        # often using `plot_model(final_model, plot='feature')` and parsing the plot data,
        # or inspecting model.feature_importances_ for tree-based models.
        # For general models, more advanced techniques like SHAP/LIME are needed.
        
        # Using LLM-identified features as a base, or generating generic ones
        feature_list_for_importance = params.get('key_features_identified', 'feature_1, feature_2, feature_3').split(', ')
        # Ensure 'target_variable' is not in feature importance list
        if params['target_variable'] in feature_list_for_importance:
            feature_list_for_importance.remove(params['target_variable'])

        feature_importance_values = {
            f.strip(): round(random.uniform(0.05, 0.35), 3) # Random values for demo
            for f in feature_list_for_importance if f.strip()
        }
        # Normalize to sum to 1, or just ensure distinct values
        total_importance = sum(feature_importance_values.values())
        if total_importance > 0:
            feature_importance_values = {k: v / total_importance for k, v in feature_importance_values.items()}
        
        # Sort by importance
        sorted_feature_importance = dict(sorted(feature_importance_values.items(), key=lambda item: item[1], reverse=True))

        job_data['feature_importance_values'] = sorted_feature_importance


        job_data['status'] = "completed"
        job_data['progress'] = 100
        job_data['api_endpoint'] = f"https://api.automl-agent.com/predict/{job_id}"
        
        # Get final metric value from finalized model's evaluation (if available)
        final_metric_value = round(random.uniform(0.7, 0.99), 4) # Simulated
        if hasattr(final_model, 'predict'): # If it's a real model, you could predict and evaluate
            # This would require a test set, which PyCaret setup handles internally.
            # A simpler way to grab final metrics after finalize_model is via `pycaret.pull()`
            try:
                metrics_df = pycaret_module['pull']()
                if not metrics_df.empty and pycaret_module['metric_name'] in metrics_df.columns:
                    final_metric_value = round(metrics_df.loc[metrics_df.index[0], pycaret_module['metric_name']], 4)
            except Exception as metric_e:
                print(f"Could not pull final metric: {metric_e}. Using simulated value.")

        job_data['final_metric_value'] = final_metric_value
        job_data['model_training_time'] = round(time.time() - job_data['start_time'], 2) # Actual simulated total time
        print(f"AutoML Engine: Job {job_id} completed.")

    except Exception as e:
        print(f"AutoML Engine: Error during pipeline for job {job_id}: {e}")
        job_data['status'] = "failed"
        job_data['error'] = str(e)
        job_data['progress'] = 100
    finally:
        _active_automl_jobs[job_id] = job_data # Ensure final state is saved


def initiate_automl_training(job_id: str, dataset_path: str, params: dict):
    """
    Kicks off the AutoML training process. This is designed to be called by the Flask app.
    In a real system, this would typically offload to a Celery task.
    """
    # Store initial job data
    _active_automl_jobs[job_id] = {
        "status": "received",
        "progress": 0,
        "api_endpoint": None,
        "error": None,
        "start_time": time.time(),
        "prompt": params.get('original_prompt', 'N/A'),
        "dataset_path": dataset_path,
        "automl_params": params, # Store all parsed/passed parameters
        "top_models_info": [], # To be filled by PyCaret
        "selected_algorithm_name": None, # To be filled later
        "model_artifact_path": None,
        "feature_importance_values": None, # Actual feature importance values
        "final_metric_value": None,
        "model_training_time": None
    }
    
    # For this demo, we run it in the same thread.
    # In production, use Celery: `task.apply_async(args=(job_id, dataset_path, params))`
    print(f"AutoML Engine: Kicking off _run_automl_pipeline for job {job_id}")
    import threading
    thread = threading.Thread(target=_run_automl_pipeline, args=(job_id, dataset_path, params))
    thread.start()
    
    return True # Indicates successful initiation

def get_automl_training_status(job_id: str) -> dict:
    """
    Retrieves the status of an AutoML training job from the in-memory store.
    """
    return _active_automl_jobs.get(job_id)

