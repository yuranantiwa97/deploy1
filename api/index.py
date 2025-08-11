# api/index.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import app   # Flask instance dari app.py
