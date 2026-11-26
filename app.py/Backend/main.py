
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from Backend.schemas import (
    PredictionRequest,
    PredictionResponse
)

from Backend.predictor import predict_category


app = FastAPI(
    title="AgriTelugu AI",
    description="Telugu Agricultural Query Classification API",
    version="1.0.0"
)


# Frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AgriTelugu AI",
        "status": "running"
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded",
        "vectorizer": "loaded"
    }


@app.post(
    "/api/v1/prediction/classify",
    response_model=PredictionResponse
)
def classify_question(request: PredictionRequest):

    try:
        return predict_category(request.question)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the backend logs."
        ) from error
