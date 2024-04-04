#!/bin/bash

# 退出脚本，如果任何命令执行失败
set -e

# 执行 Django 命令
python3 /app/manage.py makemigrations
python3 /app/manage.py migrate
python3 /app/manage.py runserver 0.0.0.0:8000
