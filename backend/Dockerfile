# backend/Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install pip-tools (optional, but good for managing dependencies)
RUN pip install pip-tools

# Copy only the requirements file first to leverage Docker cache
COPY backend/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY backend/ .
# COPY shared/ /app/shared/ # Copy shared utilities into the backend container

# Expose the port the app runs on
EXPOSE 5000

# Run the application using Gunicorn for production or Flask's dev server for development
# For development:
CMD ["python", "app.py"]

# For production (using Gunicorn, which will be configured in deployments/gunicorn.conf.py):
# CMD ["gunicorn", "--config", "deployments/gunicorn.conf.py", "app:app"]
