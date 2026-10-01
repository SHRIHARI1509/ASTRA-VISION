from pydantic import BaseModel
from typing import Optional


class ImageValidationResponse(BaseModel):
    valid: bool
    filename: str
    format: str
    width: int
    height: int
    size_bytes: int


class ImageValidationErrorDetail(BaseModel):
    code: str
    message: str


class ImageValidationErrorResponse(BaseModel):
    valid: bool = False
    error: ImageValidationErrorDetail
