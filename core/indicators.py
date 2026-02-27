"""기술적 지표 계산 — RSI, MACD, BB, MA 등."""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from loguru import logger


def calculate_all_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """OHLCV DataFrame에 기술적 지표 컬럼을 추가.

    Args:
        df: 최소 컬럼 - date, open, high, low, close, volume
        params: 지표 파라미터 (trading-params.yaml의 indicators 섹션)

    Returns:
        지표가 추가된 DataFrame
    """
    if df.empty or len(df) < 5:
        logger.warning("Not enough data to calculate indicators")
        return df

    params = params or {}
    df = df.copy()

    # 타입 보장
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    # --- 이동평균 ---
    ma_short = params.get("ma_short", 5)
    ma_long = params.get("ma_long", 20)

    df["ma5"] = SMAIndicator(close, window=ma_short).sma_indicator()
    df["ma20"] = SMAIndicator(close, window=ma_long).sma_indicator()
    df["ma60"] = SMAIndicator(close, window=60).sma_indicator()
    df["ma120"] = SMAIndicator(close, window=min(120, len(df))).sma_indicator()
    df["ema12"] = EMAIndicator(close, window=12).ema_indicator()
    df["ema26"] = EMAIndicator(close, window=26).ema_indicator()

    # 골든크로스 / 데드크로스 시그널
    df["golden_cross"] = (df["ma5"] > df["ma20"]) & (df["ma5"].shift(1) <= df["ma20"].shift(1))
    df["dead_cross"] = (df["ma5"] < df["ma20"]) & (df["ma5"].shift(1) >= df["ma20"].shift(1))

    # --- RSI ---
    rsi_period = 14
    rsi = RSIIndicator(close, window=rsi_period)
    df["rsi"] = rsi.rsi()

    # --- MACD ---
    macd_fast = params.get("macd_fast", 12)
    macd_slow = params.get("macd_slow", 26)
    macd_signal = params.get("macd_signal", 9)
    macd = MACD(close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --- 볼린저 밴드 ---
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    bb = BollingerBands(close, window=bb_period, window_dev=bb_std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = bb.bollinger_pband()  # %B

    # --- 스토캐스틱 ---
    stoch = StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # --- ATR (Average True Range) ---
    atr = AverageTrueRange(high, low, close, window=14)
    df["atr"] = atr.average_true_range()

    # --- ADX ---
    if len(df) >= 14:
        adx = ADXIndicator(high, low, close, window=14)
        df["adx"] = adx.adx()

    # --- OBV (On Balance Volume) ---
    obv = OnBalanceVolumeIndicator(close, volume)
    df["obv"] = obv.on_balance_volume()

    # --- 거래량 지표 ---
    df["volume_ma20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / df["volume_ma20"]

    # --- 수익률 ---
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    # --- 변동성 ---
    df["volatility_20d"] = df["return_1d"].rolling(window=20).std() * np.sqrt(252)

    return df


def get_indicator_summary(df: pd.DataFrame, params: dict | None = None) -> dict:
    """최신 지표 요약 — LLM에게 전달할 간결한 딕셔너리."""
    if df.empty:
        return {}

    params = params or {}
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    rsi_oversold = params.get("rsi_oversold", 35)
    rsi_overbought = params.get("rsi_overbought", 70)
    vol_surge = params.get("volume_surge_multiplier", 2.0)

    rsi_val = latest.get("rsi", 50)
    if rsi_val <= rsi_oversold:
        rsi_signal = "과매도"
    elif rsi_val >= rsi_overbought:
        rsi_signal = "과매수"
    else:
        rsi_signal = "중립"

    macd_hist = latest.get("macd_hist", 0)
    macd_prev = prev.get("macd_hist", 0)
    if macd_hist > 0 and macd_prev <= 0:
        macd_signal = "매수전환"
    elif macd_hist < 0 and macd_prev >= 0:
        macd_signal = "매도전환"
    elif macd_hist > 0:
        macd_signal = "상승추세"
    else:
        macd_signal = "하락추세"

    vol_ratio = latest.get("volume_ratio", 1.0)
    vol_signal = "거래량급증" if vol_ratio >= vol_surge else "보통"

    bb_pct = latest.get("bb_pct", 0.5)
    if bb_pct >= 1.0:
        bb_signal = "상단돌파"
    elif bb_pct <= 0.0:
        bb_signal = "하단돌파"
    else:
        bb_signal = f"밴드내({bb_pct:.1%})"

    def _safe_int(val, default=0):
        try:
            import math
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    def _safe_round(val, ndigits=1, default=0.0):
        try:
            import math
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return default
            return round(val, ndigits)
        except (ValueError, TypeError):
            return default

    return {
        "price": _safe_int(latest.get("close", 0)),
        "change_1d": f"{latest.get('return_1d', 0) or 0:.2%}",
        "ma5": _safe_int(latest.get("ma5", 0)),
        "ma20": _safe_int(latest.get("ma20", 0)),
        "ma60": _safe_int(latest.get("ma60", 0)),
        "rsi": _safe_round(rsi_val, 1),
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "macd_hist": _safe_round(macd_hist, 2),
        "bb_signal": bb_signal,
        "volume_ratio": _safe_round(vol_ratio, 2),
        "volume_signal": vol_signal,
        "golden_cross": bool(latest.get("golden_cross", False)),
        "dead_cross": bool(latest.get("dead_cross", False)),
        "adx": _safe_round(latest.get("adx", 0), 1),
        "atr": _safe_round(latest.get("atr", 0), 1),
        "volatility_20d": f"{latest.get('volatility_20d', 0) or 0:.1%}",
    }
