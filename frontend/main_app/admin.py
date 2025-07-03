# main_app/admin.py

from django.contrib import admin
from .models import UploadedDataset # Import the new model

# Register your models here.
admin.site.register(UploadedDataset)
