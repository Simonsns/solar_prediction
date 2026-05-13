import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, status
from src.utils.logger import setup_logging
from app.config import settings
from src.services.supabase_api import SupabaseAPIService
from app.routers import router as solar_router

# Init 
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init
    app.state.supabase_svc = SupabaseAPIService(settings)
    logger.info("[SUCCESS] SupabaseAPIService initialized.")
    yield 
    logger.info("Shutting down...")

# App
app = FastAPI(
    title=settings.project_name,
    version=settings.version, 
    lifespan=lifespan,
    docs_url="/docs")

# Routers
app.include_router(solar_router)

# Heath
@app.get("/health", tags=["Monitoring"], status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check.
    """
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}