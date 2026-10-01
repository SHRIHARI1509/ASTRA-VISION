import time
import threading
from typing import List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

from app.core.config import settings
from app.models.base_adapter import ModelAdapter, ModelInferenceResult, Prediction
from app.utils.logger import logger


class CLIPAdapter(ModelAdapter):
    """Offline adapter for OpenAI CLIP (openai/clip-vit-base-patch32).
    Used exclusively for offline model comparison benchmarking in Phase 8B.
    Completely isolated from the production classification pipeline.
    """

    DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device_preference: Optional[str] = None,
    ):
        self._model_id = model_id or self.DEFAULT_MODEL_ID
        self._device_preference = device_preference or settings.DEVICE
        self._device = self._resolve_device(self._device_preference)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._initialization_time_s: Optional[float] = None

    def _resolve_device(self, preference: str) -> str:
        pref = (preference or "auto").lower()
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif pref == "cpu":
            return "cpu"
        else:
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
        if self.is_loaded:
            return

        with self._lock:
            if self.is_loaded:
                return

            t0 = time.perf_counter()
            logger.info(f"Loading CLIP comparison model '{self._model_id}' on device '{self._device}'...")

            try:
                processor = AutoProcessor.from_pretrained(self._model_id)
                model = AutoModel.from_pretrained(self._model_id)
                model.to(self._device)
                model.eval()

                self._processor = processor
                self._model = model
                self._initialization_time_s = time.perf_counter() - t0
                logger.info(
                    f"CLIP model '{self._model_id}' successfully loaded in {self._initialization_time_s:.2f}s."
                )
            except Exception as e:
                logger.error(f"Failed to initialize CLIP model '{self._model_id}': {e}", exc_info=True)
                raise RuntimeError(f"CLIP initialization failed: {str(e)}") from e

    def predict(
        self,
        image: Image.Image,
        candidate_labels: Optional[List[str]] = None,
    ) -> ModelInferenceResult:
        if not self.is_loaded:
            self.load()

        labels = candidate_labels or settings.CANDIDATE_CATEGORIES
        if not labels:
            raise ValueError("Candidate labels cannot be empty.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Use identical prompt template mapping for fair comparison
        prompts = [
            settings.PROMPT_TEMPLATES.get(lbl, f"a photo of a {lbl.lower()}")
            for lbl in labels
        ]

        t0 = time.perf_counter()
        try:
            inputs = self._processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image
                # Standard CLIP normalized probabilities across candidate text prompts
                scores_tensor = logits_per_image.softmax(dim=-1).squeeze(0)
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
            predictions.sort(key=lambda p: p.score, reverse=True)

            return ModelInferenceResult(
                model=self._model_id,
                device=self._device,
                predictions=predictions,
                inference_time_ms=inference_time_ms,
                status="success",
            )
        except Exception as e:
            logger.error(f"CLIP inference failed on '{self._model_id}': {e}", exc_info=True)
            raise RuntimeError(f"Inference failed: {str(e)}") from e
