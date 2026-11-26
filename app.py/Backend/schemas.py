
from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Agricultural question in Telugu"
    )


class CategoryPrediction(BaseModel):
    category: str
    confidence: float


class PredictionResponse(BaseModel):
    original_question: str
    cleaned_question: str
    predicted_category: str
    confidence: float
    top_predictions: List[CategoryPrediction]
