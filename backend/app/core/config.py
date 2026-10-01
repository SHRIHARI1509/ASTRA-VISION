from typing import List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "ASTRA VISION"
    ENVIRONMENT: str = "development"
    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Model & Inference Configuration
    MODEL_ID: str = "google/siglip2-base-patch16-512"
    DEVICE: str = "auto"  # "auto", "cuda", or "cpu"
    PRELOAD_MODEL: bool = True

    # Candidate Categories for Zero-Shot Classification
    CANDIDATE_CATEGORIES: List[str] = [
        "Fighter Aircraft",
        "Helicopter",
        "Tank",
        "Ship",
        "Military Vehicle",
        "Drone",
    ]

    # Centralized Zero-Shot Prompt Mapping
    PROMPT_TEMPLATES: Dict[str, str] = {
        "Fighter Aircraft": "a photo of a fighter aircraft",
        "Helicopter": "a photo of a helicopter",
        "Tank": "a photo of a tank",
        "Ship": "a photo of a ship",
        "Military Vehicle": "a photo of a military vehicle",
        "Drone": "a photo of a drone",
    }

    # Centralized Uncertainty / Low-Confidence Heuristic Operating Thresholds
    # Note: These are heuristic operating thresholds based on raw SigLIP 2 similarity scores,
    # NOT statistically calibrated probabilities.
    UNCERTAINTY_SCORE_THRESHOLD: float = 0.0100   # Minimum primary score to avoid weak-detection warning
    UNCERTAINTY_MARGIN_THRESHOLD: float = 0.0200  # Minimum separation margin between rank 1 and rank 2

    # Centralized Model-Fit Justification (Phase 7D)
    MODEL_FIT_JUSTIFICATION: Dict[str, str] = {
        "model_name": "SigLIP 2 Base",
        "model_id": "google/siglip2-base-patch16-512",
        "task": "Zero-Shot Image Classification / Candidate Matching",
        "justification": (
            "ASTRA VISION uses google/siglip2-base-patch16-512 because the operational problem "
            "requires classifying reconnaissance images against a predefined taxonomy of defence objects "
            "(such as tanks, helicopters, and aircraft). SigLIP 2 provides vision-language zero-shot "
            "alignment, allowing the system to measure similarity between visual inputs and text "
            "descriptions directly. This capability enables the baseline application to evaluate and "
            "rank candidate categories without requiring a dedicated task-specific fine-tuning "
            "pipeline or collected training datasets. The model's pairwise similarity outputs directly "
            "support ASTRA VISION's multi-candidate workflow, producing both a primary prediction and "
            "deterministic Top-3 rankings that integrate with our heuristic uncertainty checks."
        ),
    }

    # Batch Processing Configuration (Phase 8C)
    MAX_BATCH_SIZE: int = 20

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
