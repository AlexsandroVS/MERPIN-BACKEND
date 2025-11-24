import sys
import os

# Add the project directory to the sys.path
project_home = os.path.dirname(os.path.abspath(__file__))
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from src.main import app
from a2wsgi import ASGIMiddleware

application = ASGIMiddleware(app)
