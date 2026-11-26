
import pickle
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
PICKLE_DIR = BASE_DIR / "picklefiles"

MODEL_PATH = PICKLE_DIR / "mlp_better_model.pkl"
VECTORIZER_PATH = PICKLE_DIR / "tfidf_vectorizer.pkl"


def load_pickle(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Pickle file not found: {file_path}"
        )

    with open(file_path, "rb") as file:
        return pickle.load(file)


# Load trained artifacts once
model = load_pickle(MODEL_PATH)
vectorizer = load_pickle(VECTORIZER_PATH)


def predict_category(question: str):
    # Basic input normalization
    cleaned_question = " ".join(question.strip().split())

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    # Convert question into TF-IDF features
    features = vectorizer.transform([cleaned_question])

    # Verify model/vectorizer compatibility
    if features.shape[1] != model.n_features_in_:
        raise ValueError(
            "TF-IDF feature count does not match model input."
        )

    # Predict category
    predicted_category = model.predict(features)[0]

    # Get category probabilities
    probabilities = model.predict_proba(features)[0]

    # Sort probabilities in descending order
    ranked_indices = probabilities.argsort()[::-1][:3]

    top_predictions = [
        {
            "category": str(model.classes_[index]),
            "confidence": round(float(probabilities[index]), 4)
        }
        for index in ranked_indices
    ]

    return {
        "original_question": question,
        "cleaned_question": cleaned_question,
        "predicted_category": str(predicted_category),
        "confidence": round(float(max(probabilities)), 4),
        "top_predictions": top_predictions
    }
