# main_app/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import json
from .models import UploadedDataset # Import the new model

def intro_page(request):
    return render(request, 'main_app/intro_page.html')

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                form.add_error(None, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'main_app/login.html', {'form': form})

def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'main_app/register.html', {'form': form})

@login_required
def user_logout(request):
    logout(request)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'message': 'Logged out successfully.', 'redirect_url': '/'})
    return redirect('intro_page')

@login_required
def dashboard(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/dashboard.html', {'current_username': current_username})

@login_required
def new_project(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/new_project.html', {'current_username': current_username})

@login_required
def my_models(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/my_models.html', {'current_username': current_username})

@login_required
def data_management(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    
    # Fetch all datasets uploaded by the current user
    uploaded_datasets = UploadedDataset.objects.filter(user=request.user).order_by('-upload_date')
    
    return render(request, 'main_app/data_management.html', {
        'current_username': current_username,
        'uploaded_datasets': uploaded_datasets
    })

@login_required
def save_uploaded_dataset_info(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            original_filename = data.get('original_filename')
            stored_filename = data.get('stored_filename')

            if not original_filename or not stored_filename:
                return JsonResponse({"error": "Missing filename information."}, status=400)

            # Check if a dataset with this stored_filename already exists for this user
            # This prevents duplicates if the frontend sends the same info multiple times
            if UploadedDataset.objects.filter(user=request.user, stored_filename=stored_filename).exists():
                return JsonResponse({"message": "Dataset already registered for this user."}, status=200)

            UploadedDataset.objects.create(
                user=request.user,
                original_filename=original_filename,
                stored_filename=stored_filename
            )
            return JsonResponse({"message": "Dataset info saved successfully."}, status=201)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method."}, status=405)
