# Telugu Agricultural Query Classification Using Deep Learning

## Project Overview

This project develops a Telugu Agricultural Query Classification system using a Multi-Layer Perceptron (MLP) neural network and TF-IDF text feature extraction.

The system identifies the category of agricultural questions and provides classification predictions through a FastAPI backend.

## Objectives

- Process Telugu agricultural questions.
- Extract text features using TF-IDF.
- Train an MLP neural network.
- Classify agricultural queries into 10 categories.
- Deploy the trained model using FastAPI.

## Deep Learning Architecture

```text
Telugu Agricultural Dataset
          |
          v
    Text Processing
          |
          v
    TF-IDF Vectorizer
          |
          v
     Input Layer
          |
          v
  Hidden Layer (64 neurons)
          |
          v
    Output Layer
          |
          v
 Predicted Category
```

## Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | MLPClassifier |
| Hidden Layer | 64 neurons |
| Feature Extraction | TF-IDF |
| Maximum Features | 2,000 |
| N-gram Range | (1, 2) |
| Framework | Scikit-learn |
| Output Categories | 10 |

## Agricultural Categories

1. Agriculture Mechanization
2. Bio-Pesticides and Bio-Fertilizers
3. Cultural Practices
4. Fertilizer Use and Availability
5. Field Preparation
6. Nutrient Management
7. Plant Protection
8. Seeds
9. Seeds and Planting Material
10. Weed Management

## Dataset

Dataset file: `agricultural_telugu_semisynthetic_5000.csv`

The dataset contains 5,000 semi-synthetic agricultural query records, balanced across 10 categories.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TF-IDF
- MLPClassifier
- FastAPI
- Uvicorn
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
Agriculture_Text_classification/
│
├── Notebooks/
├── app.py/
│   ├── Backend/
│   │   ├── main.py
│   │   ├── predictor.py
│   │   ├── schemas.py
│   │   └── __init__.py
│   │
│   ├── Dataset/
│   └── picklefiles/
│       ├── mlp_better_model.pkl
│       └── tfidf_vectorizer.pkl
│
├── .gitignore
└── README.md
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Annapurneswaraiah/Agriculture_Text_classification.git
```

### 2. Navigate to Project

```bash
cd Agriculture_Text_classification
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install fastapi uvicorn scikit-learn==1.6.1 numpy pandas pydantic
```

### 6. Start FastAPI Backend

```bash
cd app.py
python -m uvicorn Backend.main:app --reload
```

## API Documentation

Swagger UI:

http://127.0.0.1:8000/docs

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Home |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/prediction/classify` | Query classification |

### Sample Request

```json
{
  "question": "వరి పంటకు ఏ ఎరువులు ఉపయోగించాలి?"
}
```

## Future Enhancements

- Advanced Telugu NLP
- LSTM and Transformer model experiments
- RAG-based agricultural question answering
- Gemini API integration
- Telugu voice assistant
- React frontend integration

## Author

Annapurneswaraiah

GitHub: https://github.com/Annapurneswaraiah

**Domain:** Agriculture | Deep Learning | NLP
