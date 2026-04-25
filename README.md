# ML Model API

A FastAPI service for hosting ML models. Currently supports code/language detection.

## Structure

```
ml-api/
├── main.py                  # App entry point
├── requirements.txt
├── core/
│   └── model_registry.py    # Central model loader — add new models here
├── routers/
│   └── code_detect.py       # /code-detect endpoints
│   └── (your next model).py # Add new routers here
└── models/
    └── code_detector.pkl    # ← Drop your .pkl file here
```

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Usage

```bash
# Detect if text is code + which language
curl -X POST http://localhost:8000/code-detect/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "const x = useState(null)"}'

# Check all loaded models
curl http://localhost:8000/models
```

## Adding a new model

1. Drop your `.pkl` file into `models/`
2. Register it in `core/model_registry.py`
3. Create a new router in `routers/`
4. Mount it in `main.py`
