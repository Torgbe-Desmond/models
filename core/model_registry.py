"""
core/model_registry.py

Central registry for all ML models.
To add a new model:
  1. Drop the .pkl into models/
  2. Add an entry to REGISTERED_MODELS below
  3. Access it anywhere via: registry.get("your_model_name")
"""

import pickle
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "_models"

# ── Register your models here ─────────────────────────────────────────────────
# key          → name used in registry.get("key")
# file         → filename inside the _models/ directory
# description  → shown in the /_models health endpoint
REGISTERED_MODELS = {
    "detect_programming_language": {
         "file": "code_with_language_detection.pkl",
         "description": "Detect type of programming language",
     },
    # "sentiment": {
    #     "file": "sentiment.pkl",
    #     "description": "Classifies text sentiment as positive / negative / neutral.",
    # },
    # "spam_filter": {
    #     "file": "spam_filter.pkl",
    #     "description": "Detects spam messages.",
    # },
}
# ─────────────────────────────────────────────────────────────────────────────


class ModelRegistry:
    def __init__(self):
        self._store: dict[str, Any] = {}
        self._status: dict[str, str] = {}

    def load_all(self):
        """Called once at startup — loads every registered model."""
        for name, config in REGISTERED_MODELS.items():
            path = MODELS_DIR / config["file"]
            if not path.exists():
                logger.warning(f"Model file not found: {path}. Skipping '{name}'.")
                self._status[name] = "missing"
                continue
            try:
                with open(path, "rb") as f:
                    self._store[name] = pickle.load(f)
                self._status[name] = "loaded"
                logger.info(f"Loaded model: '{name}' from {path}")
            except Exception as e:
                logger.error(f"Failed to load '{name}': {e}")
                self._status[name] = f"error: {e}"

    def get(self, name: str) -> Any:
        """Returns the loaded model object, or raises if not available."""
        if name not in self._store:
            raise RuntimeError(
                f"Model '{name}' is not loaded. "
                f"Status: {self._status.get(name, 'not registered')}"
            )
        return self._store[name]

    def status(self) -> dict:
        """Returns load status for all registered models — used by /models endpoint."""
        return {
            name: {
                "status": self._status.get(name, "not attempted"),
                "description": REGISTERED_MODELS[name]["description"],
                "file": REGISTERED_MODELS[name]["file"],
            }
            for name in REGISTERED_MODELS
        }

# Singleton — import this everywhere
registry = ModelRegistry()
