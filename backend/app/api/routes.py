from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Query, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.health import HealthResponse
from app.schemas.image import (
    ImageValidationResponse,
    ImageValidationErrorResponse,
)
from app.schemas.inference import (
    InferenceTestResponse,
    InferenceErrorResponse,
)
from app.schemas.classification import (
    ClassificationResponse,
    ClassificationErrorResponse,
    BatchClassificationResponse,
)
from app.schemas.detection import (
    DetectionResponse,
    DetectionErrorResponse,
)
from app.services.image_service import image_service, ImageValidationError
from app.services.inference_service import (
    inference_service,
    ModelUnavailableError,
    InferenceExecutionError,
)
from app.services.classification_service import (
    classification_service,
    ClassificationError,
)
from app.services.detection_service import (
    detection_service,
    DetectionModelUnavailableError,
    DetectionInferenceError,
)
from app.utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return health status of the Astra Vision API service."""
    return HealthResponse(
        status="ok",
        service="astra-vision",
    )


@router.post(
    "/images/validate",
    response_model=ImageValidationResponse,
    responses={
        400: {"model": ImageValidationErrorResponse},
        422: {"model": ImageValidationErrorResponse},
    },
    tags=["Image Validation"],
)
async def validate_image(
    image: Optional[UploadFile] = File(None),
):
    """Validate an uploaded image for format, decoding, dimensions, and RGB conversion capability."""
    if image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "valid": False,
                "error": {
                    "code": "MISSING_FILE",
                    "message": "No image file provided. Please upload an image using the 'image' field.",
                },
            },
        )

    try:
        file_bytes = await image.read()
        metadata = image_service.validate_and_inspect_image(
            file_bytes=file_bytes,
            filename=image.filename,
            content_type=image.content_type,
        )
        return ImageValidationResponse(**metadata)
    except ImageValidationError as exc:
        logger.warning(f"Image validation rejected '{image.filename}': [{exc.code}] {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "valid": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            },
        )
    except Exception as exc:
        logger.error(f"Internal error processing image '{image.filename}': {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "valid": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred during image processing.",
                },
            },
        )


@router.post(
    "/inference/test",
    response_model=InferenceTestResponse,
    responses={
        400: {"model": InferenceErrorResponse},
        500: {"model": InferenceErrorResponse},
        503: {"model": InferenceErrorResponse},
    },
    tags=["Engineering Inference"],
)
async def test_inference(
    image: Optional[UploadFile] = File(None),
):
    """Engineering endpoint: Run zero-shot classification on an uploaded image."""
    if image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "MISSING_FILE",
                    "message": "No image file provided. Please upload an image using the 'image' field.",
                }
            },
        )

    try:
        file_bytes = await image.read()
        result = inference_service.run_inference(
            file_bytes=file_bytes,
            filename=image.filename or "uploaded_image",
            content_type=image.content_type,
        )
        return InferenceTestResponse(**result)
    except ImageValidationError as exc:
        logger.warning(f"Inference rejected image '{image.filename}': [{exc.code}] {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except ModelUnavailableError as exc:
        logger.error(f"Inference model unavailable: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except InferenceExecutionError as exc:
        logger.error(f"Inference execution failed: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except Exception as exc:
        logger.error(f"Unhandled error during inference on '{image.filename}': {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during inference.",
                }
            },
        )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ClassificationErrorResponse},
        500: {"model": ClassificationErrorResponse},
        503: {"model": ClassificationErrorResponse},
    },
    tags=["Classification"],
)
async def classify_object(
    image: Optional[UploadFile] = File(None),
):
    """Core classification endpoint: validate uploaded image and classify defence object."""
    if image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "MISSING_FILE",
                    "message": "No image file provided. Please upload an image using the 'image' field.",
                }
            },
        )

    try:
        file_bytes = await image.read()
        result = classification_service.classify_image(
            file_bytes=file_bytes,
            filename=image.filename or "uploaded_image",
            content_type=image.content_type,
        )
        return ClassificationResponse(**result)
    except ImageValidationError as exc:
        logger.warning(f"Classification rejected image '{image.filename}': [{exc.code}] {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except ModelUnavailableError as exc:
        logger.error(f"Classification model unavailable: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except (InferenceExecutionError, ClassificationError) as exc:
        logger.error(f"Classification failure: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except Exception as exc:
        logger.error(f"Unhandled error during classification for '{image.filename}': {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during classification.",
                }
            },
        )


@router.post(
    "/classify/batch",
    response_model=BatchClassificationResponse,
    responses={
        400: {"model": ClassificationErrorResponse},
        500: {"model": ClassificationErrorResponse},
    },
    tags=["Classification"],
)
async def classify_batch_objects(
    images: Optional[List[UploadFile]] = File(None),
):
    """Batch classification endpoint: process multiple reconnaissance images sequentially.

    Supports partial failures. Enforces MAX_BATCH_SIZE limit.
    Uses the singleton production SigLIP 2 model.
    """
    if not images or len(images) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "EMPTY_BATCH",
                    "message": "No image files provided. Please upload at least one image using the 'images' field.",
                }
            },
        )

    if len(images) > settings.MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "BATCH_SIZE_EXCEEDED",
                    "message": f"Batch size ({len(images)}) exceeds maximum permitted limit of {settings.MAX_BATCH_SIZE} images.",
                }
            },
        )

    file_tuples = []
    for img in images:
        file_bytes = await img.read()
        file_tuples.append((file_bytes, img.filename or "uploaded_image", img.content_type))

    try:
        batch_result = classification_service.classify_batch(files=file_tuples)
        return BatchClassificationResponse(**batch_result)
    except ClassificationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except Exception as exc:
        logger.error(f"Unhandled error during batch classification: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during batch classification.",
                }
            },
        )


@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        400: {"model": DetectionErrorResponse},
        500: {"model": DetectionErrorResponse},
        503: {"model": DetectionErrorResponse},
    },
    tags=["Object Detection"],
)
async def detect_objects(
    image: Optional[UploadFile] = File(None),
    box_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional detection box threshold override"),
    text_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional text alignment threshold override"),
):
    """Phase 10: Multi-object detection with bounding boxes via Grounding DINO.

    Accepts an uploaded reconnaissance image, validates boundaries, converts to RGB,
    and returns detected defence assets with bounding boxes mapped to the production taxonomy.
    """
    if image is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "MISSING_FILE",
                    "message": "No image file provided. Please upload an image using the 'image' field.",
                }
            },
        )

    try:
        file_bytes = await image.read()
        result = detection_service.detect_objects(
            file_bytes=file_bytes,
            filename=image.filename or "uploaded_image",
            content_type=image.content_type,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return DetectionResponse(**result)
    except ImageValidationError as exc:
        logger.warning(f"Detection rejected image '{image.filename}': [{exc.code}] {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except DetectionModelUnavailableError as exc:
        logger.error(f"Detection model unavailable: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except DetectionInferenceError as exc:
        logger.error(f"Detection inference failure: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )
    except Exception as exc:
        logger.error(f"Unhandled error during object detection for '{image.filename}': {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred during object detection.",
                }
            },
        )

