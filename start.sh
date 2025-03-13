#!/bin/bash

# Start making migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Load mock
python manage.py load_mock

# Start server
python manage.py runserver
