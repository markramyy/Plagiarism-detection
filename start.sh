#!/bin/bash

./manage.py makemigrations
./manage.py migrate
./manage.py load_mock
./manage.py runserver
