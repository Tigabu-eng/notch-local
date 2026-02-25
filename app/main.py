"""FastAPI application for Company Research Mapping Tool.

This module provides the main FastAPI application instance with CORS
middleware configuration and router registration for the Notch Partners
Company Research Mapping Tool.
"""

# Load environment variables before any other imports
from dotenv import load_dotenv

load_dotenv()

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.routers.calls import router as calls_router


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API version and metadata
API_VERSION = "0.1.0"
API_TITLE = "Company Research Mapping Tool API"
API_DESCRIPTION = """
Company Research Mapping Tool API for Notch Partners.

This API provides endpoints for:
- Managing company data and leadership information
- Researching and enriching company profiles
- Tracking contact status and network relationships
- Exporting data for talent landscape mapping
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    Used for initializing and cleaning up resources like database
    connections, caches, or background tasks.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after startup is complete.
    """
    # Startup: Initialize resources
    from app.services import get_search_service

    service = get_search_service()
    logger.info(f"Search service configured: {service.is_configured}")
    if service.has_tavily:
        logger.info("Tavily API is configured")
    if service.has_serp:
        logger.info("SerpAPI is configured")
    if not service.is_configured:
        logger.warning("No search APIs configured - will use mock data")
    logger.info("Application startup complete")

    yield

    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    from app.services import get_search_service

    await get_search_service().close()
    logger.info("Search service closed")


# Create FastAPI application instance
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Configure CORS middleware
# Allow requests from Vite dev server (localhost:5173) by default
# Can be overridden via CORS_ORIGINS environment variable (comma-separated list)
_default_origins = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",  # Vite dev server (alternative)
    "http://localhost:5174",  # Vite dev server (fallback port)
    "http://127.0.0.1:5174",  # Vite dev server (fallback port)
    "http://localhost:3000",  # Alternative dev server
    "http://127.0.0.1:3000",  # Alternative dev server
]

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    ALLOWED_ORIGINS = [
        origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()
    ]
else:
    ALLOWED_ORIGINS = _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["Content-Length", "Content-Type"],
)


@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """Root endpoint returning API information.

    Returns:
        Dict containing API metadata including name, version,
        description, and available documentation URLs.
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "Company Research Mapping Tool API for Notch Partners",
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "status": "operational",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring and load balancers.

    Returns:
        Dict with status indicating the API is healthy.
    """
    return {"status": "healthy"}


# Router registration
from app.routers import companies, search

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(companies.router, prefix="/api", tags=["companies"])
app.include_router(calls_router, prefix="/api", tags=["Calls"])
