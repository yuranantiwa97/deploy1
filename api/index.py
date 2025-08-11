# api/index.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # agar app.py bisa diimport

from vercel_wsgi import handle_request
from app import app  # Flask instance di app.py

def handler(request, context):
    return handle_request(app, request, context)
