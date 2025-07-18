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

# Modified user_login to handle API requests
def user_login(request):
    if request.method == 'POST':
        try:
            # Expect JSON data from the frontend
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                # Return a JSON success response with redirect URL
                return JsonResponse({'message': 'Login successful!', 'redirect_url': '/dashboard/'}, status=200)
            else:
                # Return a JSON error response
                return JsonResponse({'message': 'Invalid username or password.'}, status=401)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON in request body.'}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'An error occurred: {str(e)}'}, status=500)
    
    # For GET requests or other methods, return a message or redirect
    # This view is primarily for API calls from the frontend now.
    return JsonResponse({'message': 'This endpoint is for API login. Please use the frontend form.'}, status=405)


# Modified user_register to handle API requests
def user_register(request):
    if request.method == 'POST':
        try:
            # Expect JSON data from the frontend
            data = json.loads(request.body)
            # UserCreationForm expects data in request.POST, so we'll adapt
            # Create a mutable QueryDict from the JSON data
            from django.http import QueryDict
            q = QueryDict('', mutable=True)
            q.update(data)
            
            form = UserCreationForm(q)
            if form.is_valid():
                user = form.save()
                login(request, user) # Log the user in immediately after registration
                # Return a JSON success response with redirect URL
                return JsonResponse({'message': 'Registration successful!', 'redirect_url': '/dashboard/'}, status=201)
            else:
                # Return JSON error with form errors
                errors = form.errors.as_json()
                return JsonResponse({'message': 'Registration failed.', 'errors': json.loads(errors)}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'message': 'Invalid JSON in request body.'}, status=400)
        except Exception as e:
            return JsonResponse({'message': f'An error occurred: {str(e)}'}, status=500)
    
    # For GET requests or other methods, return a message or redirect
    return JsonResponse({'message': 'This endpoint is for API registration. Please use the frontend form.'}, status=405)


@login_required # Re-enabled login_required
def user_logout(request):
    logout(request)
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({'message': 'Logged out successfully.', 'redirect_url': '/'}, status=200)
    return redirect('intro_page')

@login_required # Re-enabled login_required
def dashboard(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/dashboard.html', {'current_username': current_username})

@login_required # Re-enabled login_required
def new_project(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/new_project.html', {'current_username': current_username})

@login_required # Re-enabled login_required
def my_models(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    return render(request, 'main_app/my_models.html', {'current_username': current_username})

@login_required # Re-enabled login_required
def data_management(request):
    current_username = request.user.username if request.user.is_authenticated else 'Guest'
    
    # Fetch all datasets uploaded by the current user
    uploaded_datasets = UploadedDataset.objects.filter(user=request.user).order_by('-upload_date')
    
    return render(request, 'main_app/data_management.html', {
        'current_username': current_username,
        'uploaded_datasets': uploaded_datasets
    })

@login_required # Re-enabled login_required
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
