from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionScore(BaseModel):
    label: str = Field(..., description="Candidate category label")
    score: float = Field(..., description="SigLIP zero-shot model similarity score")


class InferenceTestResponse(BaseModel):
    model: str = Field(..., description="Model identifier")
    device: str = Field(..., description="Hardware device used for inference (cuda or cpu)")
    predictions: List[PredictionScore] = Field(..., description="Structured prediction scores")
    inference_time_ms: float = Field(..., description="Inference execution time in milliseconds")
    status: Optional[str] = Field("success", description="Status of the inference request")


class InferenceErrorResponse(BaseModel):
    error: dict = Field(..., description="Structured error details")
