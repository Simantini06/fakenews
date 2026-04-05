"""
predict.py
----------
FakeNewsPredictor — loads the trained pipeline and exposes a predict() method.
Used by app.py.

Expected files:
    models/pipeline.pkl   (created by train.py)
    models/meta.json      (created by train.py)

WELFake label map:
    0 = REAL
    1 = FAKE
"""

import os
import json
import joblib


class FakeNewsPredictor:

    MODEL_PATH = os.path.join("models", "pipeline.pkl")
    META_PATH  = os.path.join("models", "meta.json")

    # ✅ FIXED: WELFake uses 0=REAL, 1=FAKE
    DEFAULT_LABEL_MAP = {"0": "REAL", "1": "FAKE"}

    def __init__(self):
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{self.MODEL_PATH}'. "
                "Run train.py first to generate the model."
            )

        self.pipeline  = joblib.load(self.MODEL_PATH)
        self.meta      = self._load_meta()
        self.label_map = {
            int(k): v
            for k, v in self.meta.get("label_map", self.DEFAULT_LABEL_MAP).items()
        }

        print(f"[Predictor] Model loaded — type: {self.meta.get('model_type', 'unknown')}, "
              f"accuracy: {self.meta.get('accuracy', 'N/A')}")

    def _load_meta(self) -> dict:
        if os.path.exists(self.META_PATH):
            with open(self.META_PATH) as f:
                return json.load(f)
        return {}

    def info(self) -> dict:
        return {
            "model_type":  self.meta.get("model_type", "unknown"),
            "accuracy":    self.meta.get("accuracy", None),
            "dataset":     self.meta.get("dataset", "unknown"),
            "features":    self.meta.get("features", "unknown"),
            "label_map":   self.meta.get("label_map", self.DEFAULT_LABEL_MAP),
            "model_ready": True,
        }

    def predict(self, text: str) -> dict:
        text = text.strip()

        # proba[0] = P(class 0 = REAL), proba[1] = P(class 1 = FAKE)
        proba      = self.pipeline.predict_proba([text])[0]
        pred_int   = int(proba.argmax())
        label      = self.label_map[pred_int]
        confidence = float(proba[pred_int])

        # ✅ FIXED: correctly assign fake/real probabilities
        real_prob = float(proba[0])   # class 0 = REAL
        fake_prob = float(proba[1])   # class 1 = FAKE

        if confidence >= 0.80:
            strength = "very likely"
        elif confidence >= 0.65:
            strength = "likely"
        else:
            strength = "possibly"

        verdict = f"This article is {strength} {label.lower()} news ({confidence*100:.1f}% confidence)."

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(real_prob, 4),
            "verdict":    verdict,
        }