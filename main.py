"""
main.py — Entry point for the ML Model API.

To add a new model:
  1. Register it in core/model_registry.py
  2. Create a router in routers/
  3. Import and mount it below under "Mount routers"
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.model_registry import registry
from routers import code_detect
# from routers import sentiment       # ← uncomment when you add the next model
# from routers import spam_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Lifespan: load all models once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    registry.load_all()
    logger.info("All models ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ML Model API",
    description="Hosts ML models as REST endpoints. Built to scale — drop in new models without touching existing ones.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://live-share-frontend.vercel.app","http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
app.include_router(code_detect.router)
# app.include_router(sentiment.router)
# app.include_router(spam_filter.router)


# ── Global endpoints ──────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "ML Model API is running."}


@app.get("/models", tags=["Health"], summary="List all registered models and their load status")
def list_models():
    """
    Shows every registered model, whether it loaded successfully,
    and a short description of what it does.
    """
    return {"models": registry.status()}
