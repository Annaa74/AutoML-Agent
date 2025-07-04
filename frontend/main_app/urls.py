# main_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.intro_page, name='intro_page'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('new_project/', views.new_project, name='new_project'),
    path('my_models/', views.my_models, name='my_models'),
    path('data_management/', views.data_management, name='data_management'),
    path('profile/', views.user_profile, name='user_profile'), # New URL
    path('settings/', views.user_settings, name='user_settings'), # New URL
    path('api/logout/', views.user_logout, name='logout_api'),
    path('api/save_uploaded_dataset_info/', views.save_uploaded_dataset_info, name='save_uploaded_dataset_info_api'),
]
