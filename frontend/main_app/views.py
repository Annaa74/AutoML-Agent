# main_app/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import json
from .models import UploadedDataset, UserProfile # Import new model
from .forms import CustomUserCreationForm # Import custom form

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
        form = CustomUserCreationForm(request.POST) # Use custom form
        if form.is_valid():
            user = form.save()
            # Create a UserProfile for the new user
            UserProfile.objects.create(user=user)
            login(request, user)
            return redirect('dashboard')
        else:
            print(form.errors) # Print form errors for debugging
    else:
        form = CustomUserCreationForm() # Use custom form
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
    uploaded_datasets = UploadedDataset.objects.filter(user=request.user).order_by('-upload_date')
    return render(request, 'main_app/data_management.html', {
        'current_username': current_username,
        'uploaded_datasets': uploaded_datasets
    })

@login_required
def user_profile(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    # Fetch the user's profile, create if it doesn't exist (shouldn't happen with post_save signal)
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    return render(request, 'main_app/profile.html', {
        'current_username': current_username,
        'user': request.user,
        'profile': profile
    })

@login_required
def user_settings(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    # You would typically handle a settings form submission here
    return render(request, 'main_app/settings.html', {
        'current_username': current_username,
        'user': request.user
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

