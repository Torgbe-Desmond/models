"""
routers/code_detect.py
"""

import os
import numpy as np
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy.sparse import hstack
from core.model_registry import registry

load_dotenv()

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.75"))

router = APIRouter(prefix="/code-detect", tags=["Code Detection"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    is_code: bool
    confidence: float

class PredictProgrammingLanguage(BaseModel):
    is_code: bool
    language: str | None
    confidence: float
    lang_confidence: float | None

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=50)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


# ── Exact same function you used during training ──────────────────────────────

def extract_hand_features(texts_list):
    features = []
    for text in texts_list:
        lines = text.split('\n')
        f = {
            'avg_line_len': np.mean([len(line) for line in lines]) if lines else 0,
            'symbol_ratio': sum(c in '{}()[]:;=+-*/<>!@#$%^&|`~' for c in text) / max(len(text), 1),
            'indent_count': sum(1 for line in lines if line.startswith((' ', '\t'))),
            'has_code_keyword': int(any(k in text.lower() for k in [
                'def ', 'class ', 'function ', 'const ', 'let ', 'var ', 'if (',
                'for (', 'while ', 'return ', 'import ', 'public ', 'private '
            ]))
        }
        features.append(list(f.values()))
    return np.array(features)


# ── Prediction ────────────────────────────────────────────────────────────────

def _run_programming_language_type_prediction(text: str, threshold: float) -> dict:
    bundle = registry.get("detect_programming_language")
    code_model = bundle["code_model"]   # ✅ correct key
    lang_model = bundle["lang_model"]   # ✅ correct key
    vectorizer = bundle["vectorizer"]   # ✅ correct key

    vec = vectorizer.transform([text])
    hand = extract_hand_features([text])
    combined = hstack([vec, hand])

    # Step 1: Is it code?
    code_prob = code_model.predict_proba(combined)[0][1]
    is_code = bool(code_prob > threshold)

    if not is_code:
        return {
            "is_code": False,
            "language": None,
            "confidence": round(float(code_prob), 4),
            "lang_confidence": None,
        }

    # Step 2: Which language?
    lang_probs = lang_model.predict_proba(combined)[0]
    lang_idx = lang_probs.argmax()
    language = lang_model.classes_[lang_idx]
    lang_conf = round(float(lang_probs[lang_idx]), 4)

    return {
        "is_code": True,
        "language": language,
        "confidence": round(float(code_prob), 4),
        "lang_confidence": lang_conf,
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/language", response_model=PredictProgrammingLanguage)
def predict_language_type(req: PredictRequest):
    try:
        return _run_programming_language_type_prediction(req.text, req.threshold)
    except RuntimeError as e:
        print(e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

