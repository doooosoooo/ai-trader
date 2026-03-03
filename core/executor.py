"""주문 실행 엔진 — KIS API 또는 시뮬레이션 모드로 주문 처리."""

import json
import math
from datetime import datetime
from typing import Any

import requests
from loguru import logger


# 한국 주식 호가 단위 테이블
TICK_SIZE_TABLE = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
    (float("inf"), 1_000),
]


def get_tick_size(price: int) -> int:
    """가격대별 호가 단위를 반환."""
    for threshold, tick in TICK_SIZE_TABLE:
        if price < threshold:
            return tick
    return 1_000


def adjust_price_to_tick(price: float, direction: str = "down") -> int:
    """가격을 호가 단위에 맞게 조정.

    direction: 'down' (매수 시 내림), 'up' (매도 시 올림)
    """
    tick = get_tick_size(int(price))
    if direction == "down":
        return int(price // tick * tick)
    else:
        return int(math.ceil(price / tick) * tick)


class OrderExecutor:
    """주문 실행기 — 실전/모의 모드에 따라 KIS API 호출 또는 가상 체결."""

    def __init__(self, auth, config: dict, portfolio, safety_guard):
        self.auth = auth  # KISAuth 인스턴스
        self.config = config
        self.portfolio = portfolio
        self.safety_guard = safety_guard
        self.mode = config.get("system", {}).get("mode", "simulation")

    def execute_signal(self, signal: dict, current_prices: dict) -> list[dict]:
        """LLM 시그널을 기반으로 주문 실행.

        Returns:
            실행된 주문 결과 리스트
        """
        results = []
        actions = signal.get("actions", [])

        for action in actions:
            action_type = action.get("type", "").upper()
            if action_type == "HOLD":
                continue

            ticker = action.get("ticker", "")
            if not ticker:
                logger.warning("Action with empty ticker, skipping")
                continue

            name = action.get("name", ticker)
            ratio = action.get("ratio", 0)
            urgency = action.get("urgency", "limit")
            limit_price = action.get("limit_price")
            reason = action.get("reason", "")

            try:
                if action_type == "BUY":
                    result = self._execute_buy(
                        ticker, name, ratio, urgency, limit_price,
                        current_prices.get(ticker, 0), reason,
                        json.dumps(signal, ensure_ascii=False),
                    )
                elif action_type == "SELL":
                    result = self._execute_sell(
                        ticker, name, ratio, urgency, limit_price,
                        current_prices.get(ticker, 0), reason,
                        json.dumps(signal, ensure_ascii=False),
                    )
                else:
                    continue

                if result:
                    results.append(result)

            except Exception as e:
                logger.error(f"Order execution failed for {ticker}: {e}")
                results.append({
                    "ticker": ticker,
                    "action": action_type,
                    "status": "FAILED",
                    "error": str(e),
                })

        return results

    def _execute_buy(
        self,
        ticker: str,
        name: str,
        ratio: float,
        urgency: str,
        limit_price: float | None,
        current_price: float,
        reason: str,
        signal_json: str,
    ) -> dict | None:
        if current_price <= 0:
            logger.error(f"Invalid current price for {ticker}: {current_price}")
            return None

        # 주문 금액 계산
        total_asset = self.portfolio.total_asset
        order_amount = total_asset * ratio

        # Safety: 시장가 한도 초과 시 지정가 강제
        if self.safety_guard.should_use_limit_order(order_amount):
            urgency = "limit"

        # 수량 계산
        price = limit_price or current_price
        price = adjust_price_to_tick(price, "down")
        quantity = int(order_amount / price)

        if quantity <= 0:
            logger.info(f"Calculated quantity is 0 for {ticker}, skipping")
            return None

        actual_amount = quantity * price
        fee = self._calculate_fee(actual_amount, "buy")

        # 확인 필요 여부 체크
        if self.safety_guard.needs_confirmation(actual_amount):
            logger.info(f"Order needs confirmation: {name}({ticker}) {actual_amount:,.0f}원")
            return {
                "ticker": ticker,
                "name": name,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "amount": actual_amount,
                "status": "PENDING_CONFIRMATION",
                "reason": reason,
            }

        if self.mode == "simulation":
            return self._simulate_buy(ticker, name, quantity, price, fee, reason, signal_json)
        else:
            return self._live_buy(ticker, name, quantity, price, urgency, fee, reason, signal_json)

    def _execute_sell(
        self,
        ticker: str,
        name: str,
        ratio: float,
        urgency: str,
        limit_price: float | None,
        current_price: float,
        reason: str,
        signal_json: str,
    ) -> dict | None:
        if ticker not in self.portfolio.positions:
            logger.warning(f"No position to sell: {ticker}")
            return None

        pos = self.portfolio.positions[ticker]
        quantity = int(pos.quantity * ratio)
        if quantity <= 0:
            quantity = pos.quantity  # 최소 1주

        price = limit_price or current_price
        if price <= 0:
            price = pos.current_price
        price = adjust_price_to_tick(price, "up")

        actual_amount = quantity * price
        fee = self._calculate_fee(actual_amount, "sell")

        if self.mode == "simulation":
            return self._simulate_sell(ticker, quantity, price, fee, reason, signal_json)
        else:
            return self._live_sell(ticker, quantity, price, urgency, fee, reason, signal_json)

    def _simulate_buy(self, ticker, name, quantity, price, fee, reason, signal_json) -> dict:
        """시뮬레이션 매수 — 즉시 가상 체결."""
        trade = self.portfolio.execute_buy(
            ticker=ticker, name=name, quantity=quantity,
            price=price, fee=fee, reason=reason, signal_json=signal_json,
        )
        trade["status"] = "SIMULATED"
        self.safety_guard.record_trade(ticker)
        return trade

    def _simulate_sell(self, ticker, quantity, price, fee, reason, signal_json) -> dict:
        """시뮬레이션 매도 — 즉시 가상 체결."""
        trade = self.portfolio.execute_sell(
            ticker=ticker, quantity=quantity,
            price=price, fee=fee, reason=reason, signal_json=signal_json,
        )
        trade["status"] = "SIMULATED"
        self.safety_guard.record_trade(ticker)
        return trade

    def _live_buy(self, ticker, name, quantity, price, urgency, fee, reason, signal_json) -> dict:
        """실전 매수 — KIS API 호출.

        주문 접수(SUBMITTED) 시 포트폴리오를 즉시 반영하지 않음.
        장 시작 전/후 계좌 동기화 시 실제 체결 결과로 포트폴리오 갱신.
        """
        tr_id = "TTTC0802U"  # 실전 매수
        if self.config.get("broker", {}).get("account_type") == "virtual":
            tr_id = "VTTC0802U"  # 모의 매수

        ord_type = "00"  # 지정가 (KIS: 00=지정가, 01=시장가)

        body = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "PDNO": ticker,
            "ORD_DVSN": ord_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)),
        }

        try:
            hashkey = self.auth.get_hashkey(body)
            headers = self.auth.build_headers(tr_id, hashkey)
            url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            resp = requests.post(url, json=body, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0":
                order_no = data.get("output", {}).get("ODNO", "")
                self.safety_guard.record_trade(ticker)
                logger.info(f"Live BUY order submitted: {ticker} x{quantity} @{price} (주문번호: {order_no})")
                return {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "name": name,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "amount": quantity * price,
                    "fee": fee,
                    "reason": reason,
                    "status": "SUBMITTED",
                    "order_no": order_no,
                }
            else:
                error_msg = data.get("msg1", "Unknown error")
                logger.error(f"Live BUY failed: {error_msg}")
                return {
                    "ticker": ticker, "name": name, "action": "BUY",
                    "status": "FAILED", "error": error_msg,
                }

        except Exception as e:
            logger.error(f"Live BUY exception: {e}")
            return {"ticker": ticker, "name": name, "action": "BUY", "status": "FAILED", "error": str(e)}

    def _live_sell(self, ticker, quantity, price, urgency, fee, reason, signal_json) -> dict:
        """실전 매도 — KIS API 호출.

        주문 접수(SUBMITTED) 시 포트폴리오를 즉시 반영하지 않음.
        """
        tr_id = "TTTC0801U"  # 실전 매도
        if self.config.get("broker", {}).get("account_type") == "virtual":
            tr_id = "VTTC0801U"  # 모의 매도

        ord_type = "00"  # 지정가 (KIS: 00=지정가, 01=시장가)

        # 포지션 이름 가져오기
        pos = self.portfolio.positions.get(ticker)
        name = pos.name if pos else ticker

        body = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "PDNO": ticker,
            "ORD_DVSN": ord_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(int(price)),
        }

        try:
            hashkey = self.auth.get_hashkey(body)
            headers = self.auth.build_headers(tr_id, hashkey)
            url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            resp = requests.post(url, json=body, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0":
                order_no = data.get("output", {}).get("ODNO", "")
                self.safety_guard.record_trade(ticker)
                logger.info(f"Live SELL order submitted: {ticker} x{quantity} @{price} (주문번호: {order_no})")
                # PnL 계산 (포지션 매입가 기준)
                pnl = None
                pnl_pct = None
                if pos and pos.avg_price > 0:
                    pnl = (price - pos.avg_price) * quantity - fee
                    pnl_pct = f"{(price - pos.avg_price) / pos.avg_price:.2%}"
                return {
                    "timestamp": datetime.now().isoformat(),
                    "ticker": ticker,
                    "name": name,
                    "action": "SELL",
                    "quantity": quantity,
                    "price": price,
                    "amount": quantity * price,
                    "fee": fee,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                    "status": "SUBMITTED",
                    "order_no": order_no,
                }
            else:
                error_msg = data.get("msg1", "Unknown error")
                logger.error(f"Live SELL failed: {error_msg}")
                return {
                    "ticker": ticker, "name": name, "action": "SELL",
                    "status": "FAILED", "error": error_msg,
                }

        except Exception as e:
            logger.error(f"Live SELL exception: {e}")
            return {"ticker": ticker, "name": name, "action": "SELL", "status": "FAILED", "error": str(e)}

    def _calculate_fee(self, amount: float, side: str) -> float:
        """수수료 계산 (증권사 기본 수수료율).

        매수: 0.015% (증권사 수수료)
        매도: 0.015% + 0.18% (증권거래세) + 0.02% (농어촌특별세)
        """
        broker_fee_rate = 0.00015
        if side == "sell":
            tax_rate = 0.0018 + 0.0002  # 증권거래세 + 농어촌특별세
            return amount * (broker_fee_rate + tax_rate)
        return amount * broker_fee_rate
