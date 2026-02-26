"""ML용 피처 엔지니어링."""

import numpy as np
import pandas as pd
from loguru import logger


# ML 모델에 사용할 피처 컬럼 목록
FEATURE_COLUMNS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "stoch_k", "stoch_d",
    "adx", "atr",
    "volume_ratio",
    "return_1d", "return_5d", "return_20d",
    "volatility_20d",
    "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "price_position",
]


def engineer_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """지표가 계산된 DataFrame에 ML용 피처를 추가.

    Args:
        df: calculate_all_indicators()를 통과한 DataFrame
        window: 추가 피처 계산 윈도우
    """
    if df.empty or len(df) < window:
        return df

    df = df.copy()

    # 이동평균 대비 가격 위치
    if "ma5" in df.columns:
        df["ma5_ratio"] = df["close"] / df["ma5"] - 1
    if "ma20" in df.columns:
        df["ma20_ratio"] = df["close"] / df["ma20"] - 1
    if "ma60" in df.columns:
        df["ma60_ratio"] = df["close"] / df["ma60"] - 1

    # 가격 위치 (최근 N일 범위 내 위치)
    rolling_high = df["high"].rolling(window=window).max()
    rolling_low = df["low"].rolling(window=window).min()
    price_range = rolling_high - rolling_low
    df["price_position"] = np.where(
        price_range > 0,
        (df["close"] - rolling_low) / price_range,
        0.5,
    )

    # 거래량 변화율
    df["volume_change"] = df["volume"].pct_change()

    # 갭 (전일 종가 대비 시가 차이)
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # 캔들 패턴 피처
    body = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    df["body_ratio"] = np.where(total_range > 0, body / total_range, 0)
    df["upper_shadow"] = np.where(
        total_range > 0,
        (df["high"] - df[["close", "open"]].max(axis=1)) / total_range,
        0,
    )
    df["lower_shadow"] = np.where(
        total_range > 0,
        (df[["close", "open"]].min(axis=1) - df["low"]) / total_range,
        0,
    )
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)

    return df


def create_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> pd.DataFrame:
    """ML 학습용 라벨 생성.

    Args:
        df: 피처가 포함된 DataFrame
        horizon: 예측 기간 (일)
        threshold: 상승/하락 판단 기준

    Labels:
        0 = 하락 (future return < -threshold)
        1 = 횡보 (-threshold <= future return <= threshold)
        2 = 상승 (future return > threshold)
    """
    df = df.copy()

    # N일 후 수익률
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

    # 라벨
    df["label"] = 1  # 횡보 기본값
    df.loc[df["future_return"] > threshold, "label"] = 2   # 상승
    df.loc[df["future_return"] < -threshold, "label"] = 0  # 하락

    return df


def prepare_ml_dataset(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """ML 학습/추론용 데이터셋 준비.

    Returns:
        (X, y) — y는 label이 없으면 None
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    available_cols = [c for c in feature_cols if c in df.columns]

    if not available_cols:
        logger.warning("No feature columns available")
        return pd.DataFrame(), None

    X = df[available_cols].copy()

    # NaN 처리
    X = X.ffill().fillna(0)

    # inf 처리
    X = X.replace([np.inf, -np.inf], 0)

    y = df["label"] if "label" in df.columns else None

    return X, y
