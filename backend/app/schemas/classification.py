from typing import List, Optional
from pydantic import BaseModel, Field


class PrimaryPrediction(BaseModel):
    label: str = Field(..., description="Primary classified category label")
    score: float = Field(..., description="Model similarity score for primary prediction")


class CandidateScore(BaseModel):
    label: str = Field(..., description="Candidate category label")
    score: float = Field(..., description="Model similarity score")


class InferenceMetadata(BaseModel):
    model: str = Field(..., description="Identifier of the vision model used")
    device: str = Field(..., description="Hardware execution device (e.g. cpu or cuda)")
    inference_time_ms: float = Field(..., description="Model execution time in milliseconds")


class UncertaintyDetails(BaseModel):
    is_uncertain: bool = Field(..., description="Whether the prediction is classified as uncertain under heuristic checks")
    reason: Optional[str] = Field(None, description="Controlled reason code: LOW_PRIMARY_SCORE, LOW_SCORE_MARGIN, BOTH, or None")
    method: str = Field("heuristic", description="Method used to assess uncertainty (heuristic)")
    score_margin: Optional[float] = Field(None, description="Difference between rank 1 and rank 2 candidate scores")
    primary_score: Optional[float] = Field(None, description="Raw model score of the primary prediction")
    margin_threshold: float = Field(..., description="Heuristic threshold applied for score margin")
    score_threshold: float = Field(..., description="Heuristic threshold applied for primary score")


class ModelFitJustification(BaseModel):
    model_name: str = Field(..., description="Human-readable model name")
    model_id: str = Field(..., description="Official Hugging Face model identifier")
    task: str = Field(..., description="Task for which the model is utilized")
    justification: str = Field(..., description="Concise written justification of model suitability")


class ClassificationResponse(BaseModel):
    prediction: PrimaryPrediction = Field(..., description="Highest-scoring primary prediction")
    candidates: List[CandidateScore] = Field(..., description="Deterministically ranked candidate categories and scores")
    top_3: Optional[List[CandidateScore]] = Field(None, description="Top-3 highest ranked candidate predictions")
    uncertainty: Optional[UncertaintyDetails] = Field(None, description="Heuristic uncertainty evaluation")
    model_fit: Optional[ModelFitJustification] = Field(None, description="Concise model-fit justification")
    inference: InferenceMetadata = Field(..., description="Inference execution metadata")
    status: Optional[str] = Field("success", description="Classification pipeline status")


class ClassificationErrorResponse(BaseModel):
    error: dict = Field(..., description="Structured error details")


class BatchItemSuccess(BaseModel):
    filename: str = Field(..., description="Name of evaluated image file")
    status: str = Field("success", description="Classification status for this item")
    prediction: PrimaryPrediction = Field(..., description="Highest-scoring primary prediction")
    candidates: List[CandidateScore] = Field(..., description="Deterministically ranked candidate categories")
    top_3: Optional[List[CandidateScore]] = Field(None, description="Top-3 candidate predictions")
    uncertainty: Optional[UncertaintyDetails] = Field(None, description="Heuristic uncertainty evaluation")
    model_fit: Optional[ModelFitJustification] = Field(None, description="Model-fit justification")
    inference: InferenceMetadata = Field(..., description="Inference execution metadata")


class BatchItemError(BaseModel):
    filename: str = Field(..., description="Name of rejected image file")
    status: str = Field("error", description="Failure status for this item")
    error: dict = Field(..., description="Structured error details with code and message")


class BatchClassificationResponse(BaseModel):
    status: str = Field("success", description="Overall batch response status")
    total: int = Field(..., description="Total number of items in batch")
    successful: int = Field(..., description="Number of successfully classified images")
    failed: int = Field(..., description="Number of rejected or failed images")
    results: List[dict] = Field(..., description="List of per-image classification or error results")

