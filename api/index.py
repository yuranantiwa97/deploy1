# api/index.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # agar app.py bisa diimport
from app import app  # <- WSGI application (Flask)
