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

    def __init__(self, auth, config: dict, portfolio, safety_guard, market_client=None):
        self.auth = auth  # KISAuth 인스턴스
        self.config = config
        self.portfolio = portfolio
        self.safety_guard = safety_guard
        self.market_client = market_client  # 미체결 주문 조회용
        self.mode = config.get("system", {}).get("mode", "simulation")

    def _auto_classify(self, ticker: str) -> str | None:
        """PER/PBR 기반 자동 strategy_type 분류.

        - PER 0~15 + PBR 0~1.2 → value (가치주)
        - PER -50~30 + 거래량 폭발 → daytrading (테마/모멘텀)
        - 그 외 → swing
        """
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).parent.parent / "data" / "storage" / "trader.db"
        try:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT per, pbr, change_pct, volume FROM current_prices WHERE ticker=?",
                    (ticker,),
                ).fetchone()
            if not row:
                return None
            per, pbr, change_pct, volume = row
            per = per or 0
            pbr = pbr or 0
            change_pct = change_pct or 0

            # 가치주: 저PER + 저PBR
            if 0 < per <= 15 and 0 < pbr <= 1.2:
                return "value"
            # 단타: 당일 +5% 이상 급등 (모멘텀)
            if change_pct >= 5.0:
                return "daytrading"
            # 그 외: 스윙
            return "swing"
        except Exception:
            return None

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
            strategy_type = action.get("strategy_type", "swing")

            try:
                if action_type == "BUY":
                    result = self._execute_buy(
                        ticker, name, ratio, urgency, limit_price,
                        current_prices.get(ticker, 0), reason,
                        json.dumps(signal, ensure_ascii=False),
                        strategy_type=strategy_type,
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
        strategy_type: str = "swing",
    ) -> dict | None:
        # 시초가 30분 매수 차단 (단타 제외) — 시초가 변동성 함정 회피
        from datetime import time as dtime
        now_t = datetime.now().time()
        if strategy_type != "daytrading" and dtime(9, 0) <= now_t < dtime(9, 30):
            logger.info(f"BUY blocked: {name}({ticker}) — 시초가 30분 매수 금지 ({strategy_type})")
            return None

        # 코드 기반 자동 분류 (LLM 분류 보정)
        try:
            corrected = self._auto_classify(ticker)
            if corrected and corrected != strategy_type:
                logger.info(f"strategy_type 보정: {name}({ticker}) {strategy_type} → {corrected}")
                strategy_type = corrected
        except Exception as e:
            logger.debug(f"Auto-classify failed for {ticker}: {e}")

        # 실시간 현재가 조회 (분석 시점 가격과 괴리 방지)
        if self.mode != "simulation" and self.market_client:
            try:
                live = self.market_client.get_current_price(ticker)
                live_price = live.get("price", 0)
                if live_price > 0:
                    current_price = live_price
            except Exception as e:
                logger.warning(f"Live price fetch failed for {ticker}, using analysis price: {e}")

        if current_price <= 0:
            logger.error(f"Invalid current price for {ticker}: {current_price}")
            return None

        # 5분봉 양봉 확인 (단타 제외) — 떨어지는 칼날 차단
        if strategy_type != "daytrading" and self.mode != "simulation":
            try:
                bars = self.market_client.get_minute_ohlcv(ticker, interval="5")
                if bars and len(bars) >= 1:
                    last = bars[0]
                    if last["close"] < last["open"]:
                        logger.info(f"BUY blocked: {name}({ticker}) — 5분봉 음봉 (O:{last['open']:,} > C:{last['close']:,})")
                        return None
            except Exception as e:
                logger.debug(f"Minute bar check failed for {ticker}: {e}")

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

        self._current_strategy_type = strategy_type
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

        # 라이브 모드: 미체결 매도 주문이 있으면 중복 매도 방지
        if self.mode != "simulation" and self.market_client:
            try:
                orders = self.market_client.get_today_orders()
                pending_sell_qty = sum(
                    o["ord_qty"] - o["ccld_qty"]
                    for o in orders
                    if o["ticker"] == ticker and o["side"] == "매도" and not o["filled"]
                )
                if pending_sell_qty > 0:
                    available = pos.quantity - pending_sell_qty
                    if available <= 0:
                        logger.warning(f"Already pending sell {pending_sell_qty} for {ticker}, skipping")
                        return None
                    if quantity > available:
                        logger.warning(f"Adjusting sell qty: {quantity} → {available} (pending: {pending_sell_qty})")
                        quantity = available
            except Exception as e:
                logger.warning(f"Pending order check failed, proceeding with portfolio qty: {e}")

        # 실시간 현재가 조회 (분석 시점 가격과 괴리 방지)
        if self.mode != "simulation" and self.market_client:
            try:
                live = self.market_client.get_current_price(ticker)
                live_price = live.get("price", 0)
                if live_price > 0:
                    current_price = live_price
            except Exception as e:
                logger.warning(f"Live price fetch failed for {ticker}, using analysis price: {e}")

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
            strategy_type=getattr(self, '_current_strategy_type', 'swing'),
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

        # KIS: 00=지정가, 01=시장가. urgency="market"일 때만 시장가 사용.
        is_market = urgency == "market"
        ord_type = "01" if is_market else "00"
        ord_unpr = "0" if is_market else str(int(price))

        body = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "PDNO": ticker,
            "ORD_DVSN": ord_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": ord_unpr,
        }

        try:
            url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            for attempt in range(2):
                hashkey = self.auth.get_hashkey(body)
                headers = self.auth.build_headers(tr_id, hashkey)
                resp = requests.post(url, json=body, headers=headers, timeout=10)
                if resp.status_code in (401, 403) and attempt == 0:
                    logger.warning(f"Live BUY {resp.status_code}, refreshing token...")
                    self.auth.invalidate_token()
                    import time; time.sleep(1)
                    continue
                resp.raise_for_status()
                break
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

        # KIS: 00=지정가, 01=시장가. urgency="market"일 때만 시장가 사용.
        is_market = urgency == "market"
        ord_type = "01" if is_market else "00"
        ord_unpr = "0" if is_market else str(int(price))

        # 포지션 이름 가져오기
        pos = self.portfolio.positions.get(ticker)
        name = pos.name if pos else ticker

        body = {
            "CANO": self.auth.account_no[:8],
            "ACNT_PRDT_CD": self.auth.account_product_code,
            "PDNO": ticker,
            "ORD_DVSN": ord_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": ord_unpr,
        }

        try:
            url = f"{self.auth.base_url}/uapi/domestic-stock/v1/trading/order-cash"
            for attempt in range(2):
                hashkey = self.auth.get_hashkey(body)
                headers = self.auth.build_headers(tr_id, hashkey)
                resp = requests.post(url, json=body, headers=headers, timeout=10)
                if resp.status_code in (401, 403) and attempt == 0:
                    logger.warning(f"Live SELL {resp.status_code}, refreshing token...")
                    self.auth.invalidate_token()
                    import time; time.sleep(1)
                    continue
                resp.raise_for_status()
                break
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

    def cancel_unfilled_order(
        self,
        order: dict,
        current_price: int,
        resubmit: bool = False,
    ) -> dict:
        """미체결 주문 취소 (+ 선택적 재주문).

        Args:
            order: get_today_orders() 반환 dict
            current_price: 현재가
            resubmit: True이면 취소 후 현재가로 재주문
        """
        if self.mode == "simulation":
            logger.info(f"[SIM] Would cancel {order['side']} {order['name']}({order['ticker']}) "
                        f"#{order['order_no']}")
            return {"action": "cancelled", "order": order, "simulated": True}

        remaining = order["ord_qty"] - order["ccld_qty"]
        if remaining <= 0:
            return {"action": "already_filled", "order": order}

        result = self.market_client.cancel_order(
            order_no=order["order_no"],
            order_qty=remaining,
            ticker=order["ticker"],
        )

        if not result["success"]:
            return {"action": "failed", "order": order, "error": result["message"]}

        if resubmit and current_price > 0:
            import time as _time
            _time.sleep(0.5)

            if order["side"] == "매수":
                price = adjust_price_to_tick(current_price, "down")
                new_result = self._live_buy(
                    ticker=order["ticker"], name=order["name"],
                    quantity=remaining, price=price, urgency="limit",
                    fee=self._calculate_fee(remaining * price, "buy"),
                    reason=f"[재주문] #{order['order_no']} 취소 후 재접수",
                    signal_json="",
                )
            else:
                price = adjust_price_to_tick(current_price, "up")
                new_result = self._live_sell(
                    ticker=order["ticker"], quantity=remaining,
                    price=price, urgency="limit",
                    fee=self._calculate_fee(remaining * price, "sell"),
                    reason=f"[재주문] #{order['order_no']} 취소 후 재접수",
                    signal_json="",
                )
            return {"action": "resubmitted", "order": order, "new_order": new_result, "new_price": price}

        return {"action": "cancelled", "order": order}

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
