# AutoML Agent

A full-stack web application for automated machine learning (AutoML) projects. This project combines a Django frontend with a Flask backend to provide a seamless experience for dataset management, model training, and deployment using state-of-the-art AutoML tools.

## Features

- **User Authentication:** Register, login, and manage user profiles.
- **Dataset Management:** Upload, analyze, and delete datasets.
- **AutoML Training:** Initiate and monitor model training jobs using a natural language prompt and dataset.
- **Model Management:** View, deploy, and delete trained models.
- **Frontend:** Built with Django, using templates for dashboard, project management, and settings.
- **Backend:** Flask API for handling AutoML tasks, file storage, and model artifact management.
- **Integration:** Django communicates with Flask via HTTP requests for training and status updates.

## Project Structure

```
AutoML/
├── backend/
│   ├── app.py                # Flask API server
│   ├── automl_engine.py      # AutoML logic (PyCaret, etc.)
│   ├── data_analyzer.py      # Data analysis utilities
│   ├── llm_parser.py         # LLM prompt parsing
│   ├── requirements.txt      # Backend dependencies
│   ├── uploads/              # Uploaded datasets
│   ├── trained_models/       # Model artifacts
│   └── ...
├── frontend/
│   ├── manage.py             # Django project entry
│   ├── automl_project/       # Django project settings
│   ├── main_app/             # Main Django app (views, models, templates)
│   ├── db.sqlite3            # SQLite database
│   └── ...
├── shared/                   # Shared utilities and schemas
├── deployments/              # Docker and deployment configs
└── README.md                 # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.11 (required for PyCaret compatibility)
- pip (Python package manager)

### 1. Clone the Repository
```bash
# Clone your fork or the main repo
git clone https://github.com/Annaa74/AutoML-Agent.git
cd AutoML-Agent
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv py311_env
.\py311_env\Scripts\activate
```

### 3. Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Install Frontend Dependencies
```bash
pip install django djangorestframework
```

### 5. Run Flask Backend
```bash
cd backend
python app.py
# Flask API runs on http://127.0.0.1:5000/
```

### 6. Run Django Frontend
```bash
cd ../frontend
python manage.py migrate
python manage.py runserver
# Django runs on http://127.0.0.1:8000/
```

## Usage
- Access the web app at [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Register or log in to your account
- Upload datasets, start new AutoML projects, and monitor training progress
- View, deploy, or delete your trained models

## API Endpoints

### Flask Backend
- `POST /api/train` — Start a new training job
- `GET /api/status/<job_id>` — Get status of a training job
- `POST /api/data_management/upload` — Upload a dataset
- `DELETE /api/data_management/delete/<filename>` — Delete a dataset
- `DELETE /api/model_artifacts/delete/<artifact_filename>` — Delete a model artifact

### Django Frontend
- `/api/login/` — User login (POST)
- `/api/register/` — User registration (POST)
- `/api/initiate_training/` — Initiate AutoML training (POST)
- `/api/get_training_status/<job_id>/` — Get training status (GET)

## Notes
- Make sure both Flask and Django servers are running for full functionality.
- The backend uses PyCaret, which only supports Python 3.9–3.11.
- For production, use a WSGI server (e.g., Gunicorn) and configure CORS and security settings properly.

## License
MIT License

## Author
- [Ananya  , Devansh , Aditya , Advita]
- [https://github.com/Annaa74/AutoML-Agent](https://github.com/Annaa74/AutoML-Agent)

