from typing import List, Optional, Dict, Any, Tuple

from app.core.config import settings
from app.services.inference_service import (
    InferenceService,
    inference_service,
    ModelUnavailableError,
    InferenceExecutionError,
)
from app.services.image_service import ImageValidationError
from app.utils.logger import logger


class ClassificationError(Exception):
    """Custom exception raised when classification pipeline fails."""

    def __init__(self, code: str, message: str, status_code: int = 500):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ClassificationService:
    """Core domain service for object recognition and deterministic candidate classification.

    Decoupled from ML model mechanics by delegating directly to InferenceService.
    Does not instantiate or directly reference SigLIP or any concrete ML framework.
    """

    def __init__(self, inference_svc: Optional[InferenceService] = None):
        self._inference_service = inference_svc or inference_service

    def classify_image(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_image",
        content_type: Optional[str] = None,
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify an image through the ASTRA VISION core classification engine.

        1. Delegates image validation and model inference to the InferenceService.
        2. Validates inference output structure and candidate scores.
        3. Deterministically ranks candidates with explicit tie-handling.
        4. Selects the highest-scoring candidate as the primary prediction.
        5. Returns structured domain-level classification results.

        Args:
            file_bytes: Raw binary content of the uploaded image.
            filename: Original filename.
            content_type: MIME type reported in upload header.
            candidate_labels: Optional override for candidate categories; defaults
                              to centralized settings.CANDIDATE_CATEGORIES.

        Returns:
            Structured dictionary matching ClassificationResponse schema.

        Raises:
            ImageValidationError: If the image fails Phase 2 format/integrity validation.
            ModelUnavailableError: If the model cannot be loaded or is offline.
            InferenceExecutionError: If model inference execution errors out.
            ClassificationError: If inference results are empty or malformed.
        """
        # Centralized categories single source of truth
        labels = candidate_labels or settings.CANDIDATE_CATEGORIES
        if not labels or len(labels) == 0:
            raise ClassificationError(
                code="EMPTY_CANDIDATE_TAXONOMY",
                message="Candidate taxonomy cannot be empty.",
                status_code=400,
            )

        # 1. Execute inference through inference service (which validates image raster integrity)
        inference_result = self._inference_service.run_inference(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            candidate_labels=labels,
        )

        # 2. Validate inference output
        raw_predictions = inference_result.get("predictions")
        if not raw_predictions or not isinstance(raw_predictions, list) or len(raw_predictions) == 0:
            raise ClassificationError(
                code="INVALID_INFERENCE_RESULT",
                message="Inference engine produced empty or malformed candidate predictions.",
                status_code=500,
            )

        # 3. Deterministic candidate ranking with explicit tie-handling
        # Primary key: score descending (-score)
        # Secondary key (deterministic tie-breaker): index in centralized taxonomy settings
        # Tertiary key: alphabetical label ascending
        def _deterministic_sort_key(item: Dict[str, Any]) -> tuple:
            score = float(item.get("score", 0.0))
            label = str(item.get("label", ""))
            try:
                taxonomy_index = settings.CANDIDATE_CATEGORIES.index(label)
            except ValueError:
                taxonomy_index = len(settings.CANDIDATE_CATEGORIES)
            return (-score, taxonomy_index, label)

        ranked_candidates = sorted(raw_predictions, key=_deterministic_sort_key)

        # Format candidates with rounded raw model scores
        formatted_candidates = [
            {
                "label": str(c["label"]),
                "score": round(float(c["score"]), 4),
            }
            for c in ranked_candidates
        ]

        # 4. Primary prediction selection (highest-scoring candidate)
        top_candidate = formatted_candidates[0]
        primary_prediction = {
            "label": top_candidate["label"],
            "score": top_candidate["score"],
        }

        # 5. Extract top-3 candidates safely from deterministic ranking
        top_3 = formatted_candidates[:3]

        # 6. Heuristic uncertainty evaluation (SHOULD-HAVE Feature Phase 7C)
        # Assesses model evidence using conservative, deterministic rules based on
        # primary prediction score and separation margin against runner-up.
        top_score = primary_prediction["score"]
        second_score = formatted_candidates[1]["score"] if len(formatted_candidates) > 1 else None

        low_primary = top_score < settings.UNCERTAINTY_SCORE_THRESHOLD

        if second_score is not None:
            score_margin = round(top_score - second_score, 4)
            low_margin = score_margin < settings.UNCERTAINTY_MARGIN_THRESHOLD
        else:
            score_margin = None
            low_margin = False

        if low_primary and low_margin:
            reason = "BOTH"
            is_uncertain = True
        elif low_margin:
            reason = "LOW_SCORE_MARGIN"
            is_uncertain = True
        elif low_primary:
            reason = "LOW_PRIMARY_SCORE"
            is_uncertain = True
        else:
            reason = None
            is_uncertain = False

        uncertainty_details = {
            "is_uncertain": is_uncertain,
            "reason": reason,
            "method": "heuristic",
            "score_margin": score_margin,
            "primary_score": top_score,
            "margin_threshold": settings.UNCERTAINTY_MARGIN_THRESHOLD,
            "score_threshold": settings.UNCERTAINTY_SCORE_THRESHOLD,
        }

        # 7. Extract and preserve inference execution metadata
        inference_metadata = {
            "model": inference_result.get("model", settings.MODEL_ID),
            "device": inference_result.get("device", "unknown"),
            "inference_time_ms": float(inference_result.get("inference_time_ms", 0.0)),
        }

        logger.info(
            f"Classification complete for '{filename}': Primary [{primary_prediction['label']}] "
            f"score={primary_prediction['score']:.4f} uncertain={is_uncertain} ({reason}) "
            f"({inference_metadata['device']}) in {inference_metadata['inference_time_ms']:.2f}ms"
        )

        return {
            "prediction": primary_prediction,
            "candidates": formatted_candidates,
            "top_3": top_3,
            "uncertainty": uncertainty_details,
            "model_fit": settings.MODEL_FIT_JUSTIFICATION,
            "inference": inference_metadata,
            "status": "success",
        }

    def classify_batch(
        self,
        files: List[Tuple[bytes, str, Optional[str]]],
        candidate_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify a batch of images sequentially using the single cached production model.

        Supports partial failures: invalid images produce error items without aborting the batch.
        Enforces batch size limits and deterministic ordering.

        Args:
            files: List of (file_bytes, filename, content_type) tuples.
            candidate_labels: Optional candidate categories override.

        Returns:
            Dictionary matching BatchClassificationResponse schema.

        Raises:
            ClassificationError: If batch is empty or exceeds MAX_BATCH_SIZE.
        """
        if not files or len(files) == 0:
            raise ClassificationError(
                code="EMPTY_BATCH",
                message="No image files provided in batch.",
                status_code=400,
            )

        if len(files) > settings.MAX_BATCH_SIZE:
            raise ClassificationError(
                code="BATCH_SIZE_EXCEEDED",
                message=f"Batch size ({len(files)}) exceeds maximum permitted limit of {settings.MAX_BATCH_SIZE} images.",
                status_code=400,
            )

        results: List[Dict[str, Any]] = []
        successful_count = 0
        failed_count = 0

        for file_bytes, filename, content_type in files:
            safe_name = filename or "unnamed_image"
            try:
                res = self.classify_image(
                    file_bytes=file_bytes,
                    filename=safe_name,
                    content_type=content_type,
                    candidate_labels=candidate_labels,
                )
                results.append({
                    "filename": safe_name,
                    "status": "success",
                    "prediction": res["prediction"],
                    "candidates": res["candidates"],
                    "top_3": res.get("top_3"),
                    "uncertainty": res.get("uncertainty"),
                    "model_fit": res.get("model_fit"),
                    "inference": res["inference"],
                })
                successful_count += 1
            except ImageValidationError as exc:
                logger.warning(f"Batch item '{safe_name}' validation error: [{exc.code}] {exc.message}")
                results.append({
                    "filename": safe_name,
                    "status": "error",
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                    },
                })
                failed_count += 1
            except (ModelUnavailableError, InferenceExecutionError, ClassificationError) as exc:
                logger.error(f"Batch item '{safe_name}' classification error: [{exc.code}] {exc.message}")
                results.append({
                    "filename": safe_name,
                    "status": "error",
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                    },
                })
                failed_count += 1
            except Exception as exc:
                logger.error(f"Batch item '{safe_name}' unexpected error: {str(exc)}", exc_info=True)
                results.append({
                    "filename": safe_name,
                    "status": "error",
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred during classification.",
                    },
                })
                failed_count += 1

        return {
            "status": "success",
            "total": len(files),
            "successful": successful_count,
            "failed": failed_count,
            "results": results,
        }



# Default singleton instance
classification_service = ClassificationService()
