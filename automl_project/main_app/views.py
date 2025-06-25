# main_app/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required # New decorator for views
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
import time
import uuid

# Temporary in-memory store for simulated jobs. Will be replaced by database.
training_jobs = {}

# This view renders your main HTML frontend (the intro page).
def index_view(request):
    """
    Renders the main HTML frontend page (intro page).
    If the user is already authenticated, redirect them to the dashboard.
    """
    if request.user.is_authenticated:
        return redirect('dashboard') # Redirect to dashboard if logged in
    return render(request, 'main_app/index.html')

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """
    Handles user login via API.
    Expects JSON: {"username": "...", "password": "..."}
    """
    username = request.data.get('username')
    password = request.data.get('password')

    user = authenticate(request, username=username, password=password)

    if user is not None:
        login(request, user)
        # On successful login, return a success message and allow frontend to redirect
        return Response({"message": "Login successful!", "username": user.username, "redirect_url": "/dashboard/"}, status=status.HTTP_200_OK)
    else:
        return Response({"message": "Invalid credentials."}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def signup_view(request):
    """
    Handles user registration (signup) via API.
    Expects JSON: {"username": "...", "password": "..."}
    """
    username = request.data.get('username')
    password = request.data.get('password')

    if not username or not password:
        return Response({"message": "Username and password are required."}, status=status.HTTP_400_BAD_REQUEST)

    if User.objects.filter(username=username).exists():
        return Response({"message": "Username already exists."}, status=status.HTTP_409_CONFLICT)

    try:
        user = User.objects.create_user(username=username, password=password)
        # Optionally, log the user in immediately after signup, but for now, just tell them to login
        return Response({"message": "Account created successfully! You can now log in.", "username": user.username}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"message": f"An error occurred during signup: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    Handles user logout via API.
    """
    logout(request)
    # Return a success message and a URL to redirect to the intro page
    return Response({"message": "Logged out successfully.", "redirect_url": "/"}, status=status.HTTP_200_OK)

# --- New Views for Application Sections (MPA Approach) ---

@login_required # Ensure only logged-in users can access
def dashboard_view(request):
    """Renders the dashboard page."""
    return render(request, 'main_app/dashboard.html', {'current_username': request.user.username})

@login_required
def new_project_view(request):
    """Renders the new project page."""
    return render(request, 'main_app/new_project.html', {'current_username': request.user.username})

@login_required
def my_models_view(request):
    """Renders the my models page."""
    return render(request, 'main_app/my_models.html', {'current_username': request.user.username})

@login_required
def data_management_view(request):
    """Renders the data management page."""
    return render(request, 'main_app/data_management.html', {'current_username': request.user.username})


# API endpoints for (simulated) AutoML (remain as API views)
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def train_model_api(request):
    """
    Placeholder for initiating ML model training.
    In Sprint 2, this will forward to the Flask service.
    """
    prompt = request.data.get('prompt', 'No prompt provided')
    dataset_name = request.data.get('dataset_name', 'dummy_dataset.csv')

    job_id = f"job_sprint1_sim_{uuid.uuid4().hex[:8]}"
    
    training_jobs[job_id] = {
        "status": "initiated",
        "progress": 0,
        "api_endpoint": None,
        "error": None,
        "start_time": time.time(),
        "prompt": prompt,
        "dataset_name": dataset_name
    }
    print(f"Simulated training initiation for job: {job_id}, prompt: {prompt}, dataset: {dataset_name}")
    
    return Response({"job_id": job_id, "message": "Simulated training initiated."}, status=status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_status_api(request, job_id):
    """
    Placeholder for checking model training status.
    In Sprint 2, this will query the Flask service.
    """
    job = training_jobs.get(job_id)

    if not job:
        return Response({"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND)

    if job['status'] not in ["completed", "failed"]:
        elapsed_time = time.time() - job['start_time']
        
        if elapsed_time < 2:
            job['status'] = "preprocessing data"
            job['progress'] = 20
        elif elapsed_time < 5:
            job['status'] = "training models"
            job['progress'] = 50
        elif elapsed_time < 8:
            job['status'] = "optimizing hyperparameters"
            job['progress'] = 80
        elif elapsed_time < 10:
            job['status'] = "deploying model"
            job['progress'] = 95
        else:
            job['status'] = "completed"
            job['progress'] = 100
            job['api_endpoint'] = f"https://api.automl-agent.com/predict/{job_id}/"
            print(f"Simulated training job {job_id} completed.")
    
    return Response(job, status=status.HTTP_200_OK)
