"""ML 모델 성능 평가."""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from loguru import logger


def evaluate_direction_model(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """방향 예측 모델 평가."""
    labels = {0: "하락", 1: "횡보", 2: "상승"}

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=list(labels.values()),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_profitability(
    predictions: list[dict],
    actual_returns: list[float],
) -> dict:
    """예측 기반 가상 수익률 평가.

    predictions: [{"prediction": "상승", "confidence": 0.7}, ...]
    actual_returns: 실제 수익률 리스트
    """
    if not predictions or not actual_returns:
        return {"status": "no_data"}

    correct = 0
    total = 0
    hypothetical_return = 0.0

    for pred, ret in zip(predictions, actual_returns):
        direction = pred.get("prediction", "")
        confidence = pred.get("confidence", 0)

        if direction == "상승" and confidence > 0.6:
            # 매수 시그널 → 실제 수익률 적용
            hypothetical_return += ret
            total += 1
            if ret > 0:
                correct += 1
        elif direction == "하락" and confidence > 0.6:
            # 매도/회피 시그널
            total += 1
            if ret < 0:
                correct += 1

    hit_rate = correct / total if total > 0 else 0

    return {
        "total_signals": total,
        "correct_signals": correct,
        "hit_rate": round(hit_rate, 4),
        "hypothetical_return": round(hypothetical_return, 4),
    }
