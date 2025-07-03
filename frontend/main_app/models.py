# main_app/models.py

from django.db import models
from django.contrib.auth.models import User

class UploadedDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_datasets')
    original_filename = models.CharField(max_length=255)
    # This is the unique filename stored on the server (in the 'uploads' directory by Flask)
    stored_filename = models.CharField(max_length=255, unique=True)
    upload_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s {self.original_filename} ({self.upload_date.strftime('%Y-%m-%d %H:%M')})"

    class Meta:
        ordering = ['-upload_date'] # Order by most recent first
