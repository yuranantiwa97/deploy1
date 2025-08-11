# api/index.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # tambahkan root project ke PYTHONPATH

from app import app  # expose variable 'app' (WSGI)
