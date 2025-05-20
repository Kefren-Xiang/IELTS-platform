# app/utils/intent.py

import os
import joblib

# 加载模型（只加载一次）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "intent_semantic_model.pkl")
model_bundle = joblib.load(MODEL_PATH)
encoder = model_bundle["encoder"]
classifier = model_bundle["classifier"]

def classify_intent(text: str) -> str:
    vec = encoder.encode([text])
    return classifier.predict(vec)[0]

