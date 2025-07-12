# main_app/models.py

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

class UserProfile(models.Model):
    """
    Extends the built-in Django User model to add any additional user-specific
    settings or profile information not directly part of the User model.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    # Add any additional user profile fields here in the future, e.g.:
    # phone_number = models.CharField(max_length=20, blank=True, null=True)
    # preferred_theme = models.CharField(max_length=10, default='light', choices=[('light', 'Light'), ('dark', 'Dark')])

    def __str__(self):
        return self.user.username

class UploadedDataset(models.Model):
    """
    Stores metadata about datasets uploaded by users.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_datasets')
    original_filename = models.CharField(max_length=255)
    stored_filename = models.CharField(max_length=255, unique=True, help_text="Unique filename on storage backend (e.g., Flask uploads folder).")
    upload_date = models.DateTimeField(default=timezone.now)
    # You could add more fields here, e.g., file_size, number_of_rows, number_of_columns

    def __str__(self):
        return f"{self.original_filename} (uploaded by {self.user.username})"

class TrainedModel(models.Model):
    """
    Stores metadata about machine learning models trained and deployed by the AutoML Agent.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trained_models')
    job_id = models.CharField(max_length=100, unique=True, help_text="Unique job ID from the AutoML backend.")
    model_name = models.CharField(max_length=255, help_text="User-friendly name for the model.")
    task_type = models.CharField(max_length=50, help_text="e.g., classification, regression, time_series.")
    status = models.CharField(
        max_length=50,
        default='training',
        choices=[
            ('training', 'Training in Progress'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('active', 'Active'), # For models ready for inference
            ('inactive', 'Inactive'), # For decommissioned models
            ('awaiting model selection', 'Awaiting Model Selection'), # New status
        ]
    )
    deployed_on = models.DateTimeField(null=True, blank=True, help_text="Timestamp when the model was successfully deployed.")
    api_endpoint = models.URLField(max_length=500, null=True, blank=True, help_text="API URL for model inference.")
    model_artifact_path = models.CharField(max_length=500, null=True, blank=True, help_text="Path to the saved model artifact on storage.")
    
    # Performance metrics and details
    optimization_metric = models.CharField(max_length=50, null=True, blank=True)
    metric_value = models.FloatField(null=True, blank=True)
    
    # Store feature importance as JSON string
    # Django's JSONField is available from 3.1+, for older versions use TextField and json.dumps/loads
    feature_importance_values = models.JSONField(default=dict, blank=True, null=True) 
    
    model_training_time = models.FloatField(null=True, blank=True, help_text="Total time taken for training in seconds.")
    original_prompt = models.TextField(blank=True, help_text="The natural language prompt used to train this model.")
    error_message = models.TextField(blank=True, null=True, help_text="Error message if training failed.") # New field
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.model_name} ({self.task_type}) - {self.status}"

    class Meta:
        ordering = ['-created_at']

