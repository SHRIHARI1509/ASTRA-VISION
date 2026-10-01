import time
import threading
from typing import List, Optional, Dict
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

from app.core.config import settings
from app.models.base_adapter import ModelAdapter, ModelInferenceResult, Prediction
from app.utils.logger import logger


class ModelInitializationError(Exception):
    """Raised when model loading fails."""
    pass


class ModelInferenceError(Exception):
    """Raised when model inference execution fails."""
    pass


class SigLIP2Adapter(ModelAdapter):
    """SigLIP 2 model adapter for zero-shot vision classification."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device_preference: Optional[str] = None,
    ):
        self._model_id = model_id or settings.MODEL_ID
        self._device_preference = device_preference or settings.DEVICE
        self._device = self._resolve_device(self._device_preference)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._initialization_time_s: Optional[float] = None

    def _resolve_device(self, preference: str) -> str:
        """Resolve target device with automatic hardware detection and fallback."""
        pref = (preference or "auto").lower()
        if pref == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif pref == "cpu":
            return "cpu"
        else:  # "auto"
            return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def initialization_time_s(self) -> Optional[float]:
        return self._initialization_time_s

    def load(self) -> None:
        """Initialize model and processor once, caching them in memory."""
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            t0 = time.perf_counter()
            logger.info(f"Loading SigLIP 2 model '{self._model_id}' on device '{self._device}'...")

            try:
                # Load official Hugging Face processor
                processor = AutoProcessor.from_pretrained(self._model_id)

                # Attempt loading model on target device
                try:
                    model = AutoModel.from_pretrained(self._model_id)
                    model.to(self._device)
                    model.eval()
                except Exception as cuda_err:
                    if self._device == "cuda":
                        logger.warning(
                            f"Failed loading model on CUDA: {cuda_err}. Falling back to CPU."
                        )
                        self._device = "cpu"
                        model = AutoModel.from_pretrained(self._model_id)
                        model.to("cpu")
                        model.eval()
                    else:
                        raise cuda_err

                self._processor = processor
                self._model = model
                self._initialization_time_s = time.perf_counter() - t0
                logger.info(
                    f"SigLIP 2 model '{self._model_id}' successfully loaded on "
                    f"device '{self._device}' in {self._initialization_time_s:.2f}s."
                )
            except Exception as e:
                logger.error(f"Failed to initialize SigLIP 2 model '{self._model_id}': {e}", exc_info=True)
                raise ModelInitializationError(f"Model initialization failed: {str(e)}") from e

    def predict(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> ModelInferenceResult:
        """Execute zero-shot inference on input image against candidate labels."""
        if not self.is_loaded:
            self.load()

        labels = candidate_labels or settings.CANDIDATE_CATEGORIES
        if not labels:
            raise ModelInferenceError("Candidate labels cannot be empty.")

        # Ensure image is in RGB format for processor
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Map candidate labels to prompt strings
        prompts = [
            settings.PROMPT_TEMPLATES.get(lbl, f"a photo of a {lbl.lower()}")
            for lbl in labels
        ]

        t0 = time.perf_counter()
        try:
            # Process image and prompts through official processor
            # SigLIP 2 was pre-trained with padding="max_length"
            inputs = self._processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding="max_length",
            )

            # Move inputs to active device
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image
                # SigLIP uses pairwise sigmoid activation for prediction scores
                scores_tensor = torch.sigmoid(logits_per_image)
                if scores_tensor.dim() > 1:
                    scores_tensor = scores_tensor.squeeze(0)

                if scores_tensor.dim() == 0:
                    raw_scores = [float(scores_tensor.item())]
                else:
                    raw_scores = [float(s) for s in scores_tensor.cpu().tolist()]

            t1 = time.perf_counter()
            inference_time_ms = round((t1 - t0) * 1000.0, 2)

            predictions = [
                Prediction(label=label, score=round(score, 4))
                for label, score in zip(labels, raw_scores)
            ]
            # Sort predictions by score descending
            predictions.sort(key=lambda p: p.score, reverse=True)

            return ModelInferenceResult(
                model=self._model_id,
                device=self._device,
                predictions=predictions,
                inference_time_ms=inference_time_ms,
                status="success",
            )
        except Exception as e:
            logger.error(f"Inference execution failed on model '{self._model_id}': {e}", exc_info=True)
            raise ModelInferenceError(f"Inference failed: {str(e)}") from e
