# TaskAI Backend

This is the backend for the TaskAI project, built with Django and Django REST Framework. It allows users to create, update, delete tasks and provides AI-powered task suggestions.

---

## Features

- **Task Management**: Create, Read, Update, Delete (CRUD) operations for tasks via RESTful API endpoints.
- **AI Suggestions**: Provides intelligent suggestions to help users manage their tasks efficiently.
- **JWT Authentication**: Secure API access using JSON Web Tokens with `djangorestframework-simplejwt`.
- **CORS Support**: Configured with `django-cors-headers` to allow requests from frontend applications running on different domains or ports.

---

## Tech Stack

- Python 3.10+
- Django
- Django REST Framework
- djangorestframework-simplejwt (JWT Authentication)
- django-cors-headers (CORS handling)

---

## Getting Started

Follow these instructions to get the project running locally.

### Prerequisites

- Python 3.10 or higher installed
- Git installed

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <repo-folder>

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate      # On Windows: env\Scripts\activate

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

4. **Configuration**

   Create a .env file in the project root (optional but recommended) to store sensitive settings like SECRET_KEY, database credentials, and JWT settings.

5. **Apply migrations:**

   ```bash
   python manage.py migrate

6. **Running the Server**
   Start the development server:

   ```bash
   python manage.py runserver
   
**The API will be accessible at http://127.0.0.1:8000/**
