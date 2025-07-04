# main_app/admin.py

from django.contrib import admin
from .models import UploadedDataset, UserProfile # Import the new model

# Register your models here.
admin.site.register(UploadedDataset)
admin.site.register(UserProfile)
