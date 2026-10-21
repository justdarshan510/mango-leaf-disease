import sys
import os

# Add root directory to sys.path so app.py and assets can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Export app for Vercel WSGI Serverless Function handler
app = app
