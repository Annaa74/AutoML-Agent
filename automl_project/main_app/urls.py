# main_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'), # Serves the main HTML frontend (intro page)
    
    # API endpoints for authentication (remain as JSON responses)
    path('api/login/', views.login_view, name='api_login'),
    path('api/signup/', views.signup_view, name='api_signup'),
    path('api/logout/', views.logout_view, name='api_logout'),

    # New HTML page routes for authenticated users
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('new-project/', views.new_project_view, name='new_project'),
    path('my-models/', views.my_models_view, name='my_models'),
    path('data-management/', views.data_management_view, name='data_management'),

    # API endpoints for (simulated) AutoML (remain as JSON responses)
    path('api/train_model/', views.train_model_api, name='api_train_model'),
    path('api/model_status/<str:job_id>/', views.model_status_api, name='api_model_status'),
]
