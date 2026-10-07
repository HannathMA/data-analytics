"""
Root entry point for Smart Meter Clustering application.
Enables running `python app.py` directly from the project root.
"""

import sys
from pathlib import Path

# Ensure root directory is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app
from backend.config import SERVER_HOST, SERVER_PORT, DEBUG_MODE

app = create_app()

if __name__ == "__main__":
    print(f"[*] Starting VoltCluster Smart Meter App on http://{SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)
