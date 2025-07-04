# main_app/models.py

from django.db import models
from django.contrib.auth.models import User

class UploadedDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_datasets')
    original_filename = models.CharField(max_length=255)
    stored_filename = models.CharField(max_length=255, unique=True)
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s {self.original_filename} ({self.upload_date.strftime('%Y-%m-%d %H:%M')})"

    class Meta:
        ordering = ['-upload_date']

class UserProfile(models.Model):
    """
    Model to store additional user profile information and settings.
    Linked one-to-one with Django's built-in User model.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Add any specific settings or profile fields here in the future, e.g.:
    # preferred_theme = models.CharField(max_length=50, default='light')
    # notifications_enabled = models.BooleanField(default=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

