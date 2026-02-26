"""ML 모델 학습 — LightGBM, XGBoost 기반."""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger

from data.features.feature_engineer import FEATURE_COLUMNS

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PriceDirectionTrainer:
    """가격 방향 예측 모델 학습기."""

    def __init__(self):
        self.model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        self.feature_cols = FEATURE_COLUMNS
        self.model_path = MODELS_DIR / "price_direction.pkl"
        self.meta_path = MODELS_DIR / "price_direction_meta.json"

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """시계열 교차검증으로 모델 학습."""
        # NaN/inf 제거
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            logger.warning(f"Not enough training data: {len(X_clean)} rows")
            return {"status": "failed", "reason": "insufficient_data"}

        # TimeSeriesSplit 교차검증
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X_clean):
            X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

            self.model.fit(X_train, y_train)
            score = accuracy_score(y_val, self.model.predict(X_val))
            scores.append(score)

        # 전체 데이터로 최종 학습
        self.model.fit(X_clean, y_clean)

        # 모델 저장
        self._save()

        avg_score = np.mean(scores)
        feature_importance = dict(zip(
            X_clean.columns,
            self.model.feature_importances_.tolist(),
        ))

        result = {
            "status": "success",
            "avg_accuracy": round(avg_score, 4),
            "cv_scores": [round(s, 4) for s in scores],
            "train_size": len(X_clean),
            "feature_importance": dict(sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            "trained_at": datetime.now().isoformat(),
        }

        logger.info(f"Price direction model trained: accuracy={avg_score:.4f}")
        return result

    def _save(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self) -> bool:
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False


class AnomalyDetectorTrainer:
    """이상 탐지 모델 학습기 — Isolation Forest."""

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.model_path = MODELS_DIR / "anomaly_detector.pkl"
        self.feature_cols = ["return_1d", "volume_ratio", "volatility_20d", "bb_pct", "rsi"]

    def train(self, X: pd.DataFrame) -> dict:
        cols = [c for c in self.feature_cols if c in X.columns]
        if not cols:
            return {"status": "failed", "reason": "no_features"}

        X_clean = X[cols].dropna()
        X_clean = X_clean.replace([np.inf, -np.inf], 0)

        if len(X_clean) < 50:
            return {"status": "failed", "reason": "insufficient_data"}

        self.model.fit(X_clean)

        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"Anomaly detector trained on {len(X_clean)} rows")
        return {
            "status": "success",
            "train_size": len(X_clean),
            "features_used": cols,
            "trained_at": datetime.now().isoformat(),
        }

    def load(self) -> bool:
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False
