"""
Flask REST Application Server for Smart Meter Clustering.
Serves API endpoints and frontend dashboard.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    SERVER_HOST,
    SERVER_PORT,
    DEBUG_MODE,
    MAX_CONTENT_LENGTH,
)
from backend.api.routes import api_bp


def create_app() -> Flask:
    """Application factory for Flask backend."""
    template_dir = PROJECT_ROOT / "frontend" / "templates"
    static_dir = PROJECT_ROOT / "frontend" / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )

    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["SECRET_KEY"] = "smart-meter-acf-clustering-secret"

    # Register API blueprint
    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"[*] Starting Smart Meter Clustering server on http://{SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)
