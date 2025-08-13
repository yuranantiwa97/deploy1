# Vercel Python WSGI entry
import os, sys

# tambahkan repo root ke sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # /api
ROOT_DIR = os.path.dirname(BASE_DIR)                         # repo root
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# impor Flask app dari app.py di root
from app import app  # WSGI callable bernama "app"
