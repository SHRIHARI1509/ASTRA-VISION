"""ASTRA VISION — Phase 10 Grounding DINO Open-Vocabulary Detector Adapter.

Wraps Hugging Face transformers AutoProcessor and AutoModelForZeroShotObjectDetection
for IDEA-Research/grounding-dino-base. Provides in-memory singleton caching,
device resolution (CUDA with CPU fallback), taxonomy normalization, and validated
coordinate bounding.
"""

import time
import re
from typing import List, Dict, Any, Optional
from PIL import Image
import torch

from app.core.config import settings
from app.schemas.detection import BoxCoordinates, DetectedObject
from app.utils.logger import logger


class DetectionModelInitializationError(Exception):
    """Raised when the Grounding DINO model or processor fails to initialize."""
    def __init__(self, message: str = "Grounding DINO detector failed to initialize."):
        self.message = message
        self.code = "DETECTION_MODEL_INIT_FAILED"
        self.status_code = 503
        super().__init__(message)


class DetectionModelInferenceError(Exception):
    """Raised when Grounding DINO execution encounters an inference failure."""
    def __init__(self, message: str = "Grounding DINO detector inference failed."):
        self.message = message
        self.code = "DETECTION_INFERENCE_FAILED"
        self.status_code = 500
        super().__init__(message)


class GroundingDINOAdapter:
    """Encapsulates Grounding DINO open-vocabulary detector loading and inference."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_id = model_id or settings.DETECTION_MODEL_ID
        self._device = self._resolve_device(device or settings.DETECTION_DEVICE)
        self._processor = None
        self._model = None
        self._is_loaded: bool = False
        self._initialization_time_s: Optional[float] = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def initialization_time_s(self) -> Optional[float]:
        return self._initialization_time_s

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve requested device string to active hardware runtime."""
        if requested_device.lower() == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device.lower() == "cpu":
            return "cpu"
        else:
            return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load and cache processor and model weights in memory. Thread-safe singleton pattern."""
        if self._is_loaded:
            return

        t0 = time.time()
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            logger.info(f"Loading Grounding DINO processor for '{self._model_id}'...")
            self._processor = AutoProcessor.from_pretrained(self._model_id)

            logger.info(f"Loading Grounding DINO model '{self._model_id}' onto device '{self._device}'...")
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self._model_id)
            self._model.to(self._device)
            self._model.eval()

            self._is_loaded = True
            self._initialization_time_s = round(time.time() - t0, 3)
            logger.info(f"Grounding DINO detector ready in {self._initialization_time_s}s on {self._device}.")
        except Exception as exc:
            self._is_loaded = False
            logger.error(f"Failed to load Grounding DINO detector '{self._model_id}': {exc}", exc_info=True)
            raise DetectionModelInitializationError(
                f"Failed to load Grounding DINO detector '{self._model_id}': {str(exc)}"
            )

    @staticmethod
    def normalize_label(raw_label: str) -> Optional[str]:
        """Normalize raw detector phrase into ASTRA VISION's production taxonomy.
        
        Only returns labels that map safely to one of the 6 production classes:
        - Tank
        - Military Vehicle
        - Fighter Aircraft
        - Helicopter
        - Ship
        - Drone
        
        Returns None for unmapped or unrelated detections.
        """
        if not raw_label or not isinstance(raw_label, str):
            return None

        # Clean string: lowercase, strip punctuation and extra spaces
        cleaned = raw_label.strip().lower()
        cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()

        # Direct dictionary lookup
        if cleaned in settings.DETECTION_LABEL_MAP:
            return settings.DETECTION_LABEL_MAP[cleaned]

        # Strip leading articles ('a ', 'an ', 'the ')
        stripped = re.sub(r"^(a|an|the)\s+", "", cleaned).strip()
        if stripped in settings.DETECTION_LABEL_MAP:
            return settings.DETECTION_LABEL_MAP[stripped]

        # Check for exact token containment in candidate categories
        for cat in settings.CANDIDATE_CATEGORIES:
            cat_lower = cat.lower()
            if cat_lower == stripped or cat_lower == cleaned:
                return cat

        # Known military vehicle synonyms
        if stripped in {"tanker", "armored vehicle", "armoured vehicle", "apc", "ifv"}:
            return "Military Vehicle"

        # Known aircraft synonyms
        if stripped in {"warplane", "jet fighter", "fighter jet"}:
            return "Fighter Aircraft"

        # Known ship synonyms
        if stripped in {"warship", "destroyer", "frigate", "aircraft carrier", "naval vessel"}:
            return "Ship"

        return None

    def detect(
        self,
        image: Image.Image,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute open-vocabulary object detection on a PIL image.
        
        Args:
            image: Standardized PIL Image instance
            box_threshold: Minimum detection box score threshold
            text_threshold: Minimum text-alignment score threshold
            
        Returns:
            Dictionary matching the DetectionResponse schema.
        """
        if not self._is_loaded:
            self.load()

        threshold = box_threshold if box_threshold is not None else settings.DETECTION_BOX_THRESHOLD
        txt_threshold = text_threshold if text_threshold is not None else settings.DETECTION_TEXT_THRESHOLD

        # Ensure image is canonical 3-channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        prompt_text = " ".join(settings.DETECTION_PROMPTS)

        t0 = time.time()
        try:
            inputs = self._processor(images=image, text=prompt_text, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process detections into absolute pixel boxes
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=threshold,
                text_threshold=txt_threshold,
                target_sizes=[(orig_h, orig_w)],
            )
        except Exception as exc:
            logger.error(f"Grounding DINO inference error: {exc}", exc_info=True)
            raise DetectionModelInferenceError(f"Inference execution failed: {str(exc)}")

        inference_time_ms = round((time.time() - t0) * 1000.0, 2)

        detections: List[DetectedObject] = []
        if results and len(results) > 0:
            res = results[0]
            scores = res.get("scores", torch.tensor([])).tolist()
            # Support both text_labels (newer transformers) and labels (older transformers)
            labels = res.get("text_labels") or res.get("labels", [])
            boxes = res.get("boxes", torch.tensor([])).tolist()

            for score, raw_label, box in zip(scores, labels, boxes):
                normalized_class = self.normalize_label(str(raw_label))
                if not normalized_class:
                    # Skip detections that do not map to the production taxonomy
                    logger.debug(f"Skipping unmapped detected label: '{raw_label}'")
                    continue

                # Box coordinates [xmin, ymin, xmax, ymax]
                x1_raw, y1_raw, x2_raw, y2_raw = box

                # Clamp coordinates to image boundaries
                x1 = max(0, min(orig_w - 1, int(round(x1_raw))))
                y1 = max(0, min(orig_h - 1, int(round(y1_raw))))
                x2 = max(x1 + 1, min(orig_w, int(round(x2_raw))))
                y2 = max(y1 + 1, min(orig_h, int(round(y2_raw))))

                # Validate strict inequalities x1 < x2 and y1 < y2
                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    DetectedObject(
                        class_name=normalized_class,
                        score=round(float(score), 4),
                        box=BoxCoordinates(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

        # Sort detections descending by score
        detections.sort(key=lambda d: d.score, reverse=True)

        return {
            "detections": detections,
            "image_width": orig_w,
            "image_height": orig_h,
            "count": len(detections),
            "detector": self._model_id,
            "device": self._device,
            "inference_time_ms": inference_time_ms,
        }
