"""ML 모델 추론 — 학습된 모델로 예측 수행."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from data.features.feature_engineer import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent / "models"

DIRECTION_LABELS = {0: "하락", 1: "횡보", 2: "상승"}


class MLPredictor:
    """학습된 ML 모델로 예측 수행."""

    def __init__(self):
        self._direction_model = None
        self._anomaly_model = None
        self._load_models()

    def _load_models(self):
        direction_path = MODELS_DIR / "price_direction.pkl"
        anomaly_path = MODELS_DIR / "anomaly_detector.pkl"

        if direction_path.exists():
            with open(direction_path, "rb") as f:
                self._direction_model = pickle.load(f)
            logger.info("Price direction model loaded")

        if anomaly_path.exists():
            with open(anomaly_path, "rb") as f:
                self._anomaly_model = pickle.load(f)
            logger.info("Anomaly detector model loaded")

    def predict_direction(self, X: pd.DataFrame) -> dict:
        """가격 방향 예측.

        Returns:
            {
                "prediction": "상승" | "횡보" | "하락",
                "probabilities": {"하락": 0.2, "횡보": 0.3, "상승": 0.5},
                "confidence": 0.5
            }
        """
        if self._direction_model is None:
            return {"prediction": "unknown", "reason": "model_not_trained"}

        available_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
        if not available_cols:
            return {"prediction": "unknown", "reason": "no_features"}

        # 최신 행만 사용
        row = X[available_cols].iloc[[-1]].fillna(0).replace([np.inf, -np.inf], 0)

        try:
            pred = self._direction_model.predict(row)[0]
            proba = self._direction_model.predict_proba(row)[0]

            prediction = DIRECTION_LABELS.get(int(pred), "unknown")
            probabilities = {
                DIRECTION_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            }
            confidence = round(float(max(proba)), 4)

            return {
                "prediction": prediction,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        except Exception as e:
            logger.error(f"Direction prediction failed: {e}")
            return {"prediction": "error", "reason": str(e)}

    def detect_anomaly(self, X: pd.DataFrame) -> dict:
        """이상 탐지.

        Returns:
            {"is_anomaly": bool, "anomaly_score": float}
        """
        if self._anomaly_model is None:
            return {"is_anomaly": False, "reason": "model_not_trained"}

        anomaly_features = ["return_1d", "volume_ratio", "volatility_20d", "bb_pct", "rsi"]
        available = [c for c in anomaly_features if c in X.columns]
        if not available:
            return {"is_anomaly": False, "reason": "no_features"}

        row = X[available].iloc[[-1]].fillna(0).replace([np.inf, -np.inf], 0)

        try:
            prediction = self._anomaly_model.predict(row)[0]
            score = self._anomaly_model.decision_function(row)[0]

            return {
                "is_anomaly": prediction == -1,
                "anomaly_score": round(float(score), 4),
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"is_anomaly": False, "reason": str(e)}

    def predict_all(self, X: pd.DataFrame) -> dict:
        """모든 모델의 예측 결과를 통합."""
        return {
            "direction": self.predict_direction(X),
            "anomaly": self.detect_anomaly(X),
        }

    @property
    def is_ready(self) -> bool:
        return self._direction_model is not None
