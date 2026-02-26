"""ML 엔진 — 학습/추론 통합 관리."""

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from data.features.feature_engineer import (
    engineer_features,
    create_labels,
    prepare_ml_dataset,
)
from ml.trainer import PriceDirectionTrainer, AnomalyDetectorTrainer
from ml.predictor import MLPredictor


class MLEngine:
    """ML 모델 학습/추론 통합 관리."""

    def __init__(self, config: dict):
        self.config = config.get("ml", {})
        self.direction_trainer = PriceDirectionTrainer()
        self.anomaly_trainer = AnomalyDetectorTrainer()
        self.predictor = MLPredictor()

    def train_all(self, data: dict[str, pd.DataFrame]) -> dict:
        """모든 모델 학습.

        Args:
            data: {ticker: indicator_df} 딕셔너리
        """
        results = {}

        # 가격 방향 예측 모델
        if self.config.get("models", {}).get("price_direction", True):
            result = self._train_direction(data)
            results["price_direction"] = result

        # 이상 탐지 모델
        if self.config.get("models", {}).get("anomaly_detection", True):
            result = self._train_anomaly(data)
            results["anomaly_detection"] = result

        # 학습 후 predictor 리로드
        self.predictor = MLPredictor()

        return results

    def _train_direction(self, data: dict[str, pd.DataFrame]) -> dict:
        """가격 방향 예측 모델 학습 — 전 종목 데이터 통합."""
        all_X = []
        all_y = []

        for ticker, df in data.items():
            if df.empty or len(df) < 30:
                continue

            df = engineer_features(df)
            df = create_labels(df, horizon=5, threshold=0.02)
            X, y = prepare_ml_dataset(df)

            if X.empty or y is None:
                continue

            # label이 있는 행만
            mask = y.notna()
            all_X.append(X[mask])
            all_y.append(y[mask])

        if not all_X:
            return {"status": "failed", "reason": "no_data"}

        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)

        return self.direction_trainer.train(X_combined, y_combined)

    def _train_anomaly(self, data: dict[str, pd.DataFrame]) -> dict:
        """이상 탐지 모델 학습."""
        all_X = []

        for ticker, df in data.items():
            if df.empty or len(df) < 30:
                continue
            df = engineer_features(df)
            X, _ = prepare_ml_dataset(df)
            if not X.empty:
                all_X.append(X)

        if not all_X:
            return {"status": "failed", "reason": "no_data"}

        X_combined = pd.concat(all_X, ignore_index=True)
        return self.anomaly_trainer.train(X_combined)

    def predict(self, df: pd.DataFrame) -> dict:
        """단일 종목 예측."""
        if df.empty:
            return {"direction": {"prediction": "unknown"}, "anomaly": {"is_anomaly": False}}

        df = engineer_features(df)
        X, _ = prepare_ml_dataset(df)

        if X.empty:
            return {"direction": {"prediction": "unknown"}, "anomaly": {"is_anomaly": False}}

        return self.predictor.predict_all(X)

    @property
    def is_ready(self) -> bool:
        return self.predictor.is_ready
