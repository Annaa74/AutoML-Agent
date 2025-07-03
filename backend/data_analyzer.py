# backend/data_analyzer.py

import pandas as pd
import io
import json
import random
import numpy as np

# Import necessary libraries for BART (commented out for demo to avoid heavy downloads unless needed)
# from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
# import torch

# Initialize BART Zero-Shot Classifier (loaded once when module is imported)
# This model is good for classification without explicit training data.
# try:
#     # Using CPU for broader compatibility in Canvas environment
#     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#     print("BART 'facebook/bart-large-mnli' zero-shot classifier loaded successfully for data_analyzer.")
# except ImportError:
#     print("Transformers or PyTorch not installed correctly. BART Zero-Shot Classifier will not be available for data_analyzer.")
#     classifier = None
# except Exception as e:
#     print(f"Error loading BART Zero-Shot Classifier for data_analyzer: {e}. It will not be available.")
#     classifier = None

# You could also load a text generation/summarization model for more complex parsing
# model_name_for_generation = "facebook/bart-large-cnn"
# try:
#     tokenizer_gen = BartTokenizer.from_pretrained(model_name_for_generation)
#     model_gen = BartForConditionalGeneration.from_pretrained(model_name_for_generation)
#     generator = pipeline("text2text-generation", model=model_gen, tokenizer=tokenizer_gen)
#     print(f"BART '{model_name_for_generation}' generator loaded successfully for data_analyzer.")
# except ImportError:
#     print("Transformers or PyTorch not installed correctly. BART generator will not be available for data_analyzer.")
#     generator = None
# except Exception as e:
#     print(f"Error loading BART generator for data_analyzer: {e}. It will not be available.")
#     generator = None


def load_dataset_and_get_preview(file_path: str):
    """
    Loads a dataset from the given path and returns its head, info, and basic description.
    Supports CSV and Excel.
    """
    df = None
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    # Get head (first 5 rows)
    head_html = df.head().to_html(classes='table-auto w-full text-left whitespace-nowrap text-sm')

    # Get info (non-null counts, dtypes)
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    # Get describe (statistical summary)
    describe_html = df.describe().to_html(classes='table-auto w-full text-left whitespace-nowrap text-sm')

    return df, head_html, info_str, describe_html


def generate_dataset_summary_llm(df: pd.DataFrame, df_info_str: str, df_describe_html: str) -> str:
    """
    Generates a natural language summary of the dataset using an LLM (BART conceptually).
    """
    summary_prompt_parts = [
        "Analyze the following dataset information and provide a concise, natural language summary.",
        f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.",
        "Column types and non-null counts:",
        df_info_str,
        "Statistical summary of numerical columns:",
        df_describe_html,
        "Identify potential target variables (e.g., 'Price', 'Churn', 'Sales').",
        "Highlight any interesting patterns, potential issues (e.g., missing values, skewed distributions), or key features."
    ]
    full_prompt = "\n".join(summary_prompt_parts)

    # --- Conceptual BART Integration for Summary Generation (Commented Out) ---
    # This is how you would use a generative BART model if it were fine-tuned
    # for structured summarization or general text generation from data insights.
    # For a general BART model (like bart-large-cnn), it's primarily for summarization
    # of long texts, not direct structured data interpretation.

    # if generator:
    #     try:
    #         # A more sophisticated prompt would be needed for structured output
    #         # For general summarization:
    #         generated_summary = generator(full_prompt, max_length=250, min_length=50, do_sample=False)
    #         return generated_summary[0]['generated_text']
    #     except Exception as e:
    #         print(f"Error during BART summary generation: {e}. Falling back to simulated summary.")

    # --- Simulated Summary (Active for Demo) ---
    # This simulation provides a plausible summary without requiring a live LLM.
    summary = f"This dataset contains {df.shape[0]} rows and {df.shape[1]} columns. "
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numerical_cols:
        summary += f"It includes numerical features such as {', '.join(numerical_cols[:3])}{'...' if len(numerical_cols) > 3 else ''}. "
    if categorical_cols:
        summary += f"Categorical features include {', '.join(categorical_cols[:3])}{'...' if len(categorical_cols) > 3 else ''}. "

    # Identify potential target variables based on common names or cardinality
    potential_targets = []
    for col in df.columns:
        if df[col].nunique() < 20 and df[col].dtype in ['object', 'category', 'int64']: # Low cardinality for classification
            potential_targets.append(col)
        elif df[col].dtype in ['float64', 'int64'] and df[col].nunique() > df.shape[0] * 0.1: # High cardinality for regression
             potential_targets.append(col)

    if potential_targets:
        summary += f"Potential target variables for predictive modeling might include: {', '.join(potential_targets[:3])}{'...' if len(potential_targets) > 3 else ''}. "
    else:
        summary += "No obvious target variables were identified based on common patterns. "

    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].index.tolist()
    if missing_cols:
        summary += f"Some columns have missing values, notably: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}. "
    else:
        summary += "The dataset appears to be complete with no missing values. "

    summary += "Overall, this dataset provides a rich foundation for various machine learning tasks, depending on the specific problem you aim to solve."
    
    return summary


def generate_visualization_data(df: pd.DataFrame):
    """
    Generates data for basic visualizations (histograms for numerical, bar charts for categorical)
    suitable for Chart.js. Returns a list of chart configurations.
    """
    charts = []

    # Process numerical columns for histograms
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].nunique() > 10: # Only plot if enough unique values for a meaningful histogram
            hist_data, bin_edges = np.histogram(df[col].dropna(), bins=10)
            labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
            charts.append({
                'type': 'bar',
                'title': f'Distribution of {col}',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Frequency',
                        'data': hist_data.tolist(),
                        'backgroundColor': 'rgba(75, 192, 192, 0.6)',
                        'borderColor': 'rgba(75, 192, 192, 1)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Frequency'}},
                        'x': {'title': {'display': True, 'text': col}}
                    }
                }
            })

    # Process categorical columns for bar charts
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 20: # Limit to columns with reasonable number of unique categories
            value_counts = df[col].value_counts().sort_index()
            charts.append({
                'type': 'bar',
                'title': f'Counts of {col}',
                'data': {
                    'labels': value_counts.index.tolist(),
                    'datasets': [{
                        'label': 'Count',
                        'data': value_counts.values.tolist(),
                        'backgroundColor': 'rgba(153, 102, 255, 0.6)',
                        'borderColor': 'rgba(153, 102, 255, 1)',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'y': {'beginAtZero': True, 'title': {'display': True, 'text': 'Count'}},
                        'x': {'title': {'display': True, 'text': col}}
                    }
                }
            })
    
    return charts
