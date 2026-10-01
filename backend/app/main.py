from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.api.routes import router as api_router
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for FastAPI application."""
    logger.info(f"Initialized {settings.PROJECT_NAME} in [{settings.ENVIRONMENT}] mode.")
    if settings.PRELOAD_MODEL:
        try:
            from app.services.inference_service import inference_service
            inference_service.initialize_model()
        except Exception as e:
            logger.error(f"Error during startup model loading: {e}", exc_info=True)
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}.")


def create_app() -> FastAPI:
    """FastAPI application factory for Astra Vision."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration for frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routing
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # Clean error handling foundation
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.warning(f"HTTP {exc.status_code} error on {request.url.path}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation Error",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal Server Error",
                }
            },
        )

    return app


app = create_app()
