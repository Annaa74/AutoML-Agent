# main_app/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User

class CustomUserCreationForm(UserCreationForm):
    """
    A custom UserCreationForm to include first name, last name, and email.
    """
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=254, help_text='Required. Enter a valid email address.', required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('first_name', 'last_name', 'email',)

class CustomUserChangeForm(UserChangeForm):
    """
    A custom UserChangeForm for admin interface if needed,
    to include first name, last name, and email.
    """
    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')

