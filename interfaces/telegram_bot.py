"""Telegram 봇 — 양방향 알림 + 명령 수신."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

_PENDING_SELLS_PATH = Path(__file__).parent.parent / "data" / "pending_sell_confirmations.json"

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available")


class TelegramBot:
    """Telegram 봇 — 알림 발송 및 사용자 명령 수신."""

    def __init__(self, config: dict, system_ref=None):
        self.enabled = config.get("telegram", {}).get("enabled", True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.alert_levels = config.get("telegram", {}).get("alert_levels", [])
        self.system = system_ref  # TradingSystem 참조
        self._app = None
        self._pending_confirmations: dict[str, dict] = {}
        self._pending_param_changes: dict[str, dict] = {}  # force param 확인 대기
        self._pending_sell_confirmations: dict[str, dict] = {}  # 익절 확인 대기
        self._loop = None  # 메인 이벤트 루프 참조 (스레드에서 async 호출용)
        self._load_pending_sells()  # 재시작 시 복구

        if not TELEGRAM_AVAILABLE:
            self.enabled = False

    async def start(self):
        """봇 시작."""
        if not self.enabled or not self.token:
            logger.info("Telegram bot disabled or no token")
            return

        self._app = Application.builder().token(self.token).build()

        # 명령 핸들러 등록
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("today", self._cmd_today))
        self._app.add_handler(CommandHandler("strategy", self._cmd_strategy))
        self._app.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        self._app.add_handler(CommandHandler("pause", self._cmd_pause))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("mode", self._cmd_mode))
        self._app.add_handler(CommandHandler("mode_confirm_live", self._cmd_mode_confirm_live))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CommandHandler("setcapital", self._cmd_setcapital))
        self._app.add_handler(CommandHandler("orders", self._cmd_orders))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("analysis", self._cmd_analysis))
        self._app.add_handler(CommandHandler("screen", self._cmd_screen))
        self._app.add_handler(CommandHandler("backtest", self._cmd_backtest))
        self._app.add_handler(CommandHandler("param", self._cmd_param))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        self._loop = asyncio.get_event_loop()  # 메인 루프 참조 저장
        logger.info("Telegram bot started")

        # 재시작 시 복원된 pending sell에 대해 타임아웃 태스크 재스케줄링
        self._reschedule_pending_sell_timeouts()

    async def stop(self):
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass

    # --- 익절 확인 저장/복구 ---

    def _load_pending_sells(self):
        """재시작 시 미처리 익절 확인 복구."""
        try:
            if _PENDING_SELLS_PATH.exists():
                with open(_PENDING_SELLS_PATH, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # 24시간 이상 된 건 제거
                now = datetime.now()
                for conf_id, item in list(saved.items()):
                    req_time = datetime.fromisoformat(item.get("requested_at", ""))
                    if (now - req_time).total_seconds() > 86400:
                        del saved[conf_id]
                self._pending_sell_confirmations = saved
                if saved:
                    logger.info(f"Restored {len(saved)} pending sell confirmations")
        except Exception as e:
            logger.warning(f"Failed to load pending sells: {e}")

    def _reschedule_pending_sell_timeouts(self):
        """재시작 시 복원된 pending sell에 대해 타임아웃 태스크를 재생성."""
        if not self._pending_sell_confirmations:
            return
        timeout = self._get_confirmation_timeout()
        now = datetime.now()
        for conf_id, item in list(self._pending_sell_confirmations.items()):
            if item.get("status") != "pending":
                continue
            # 이미 경과한 시간을 계산하여 남은 타임아웃 적용
            req_time = datetime.fromisoformat(item.get("requested_at", ""))
            elapsed = (now - req_time).total_seconds()
            remaining = max(0, timeout - elapsed)
            logger.info(f"Rescheduling sell timeout for {conf_id}: {remaining:.0f}s remaining")
            asyncio.ensure_future(self._auto_sell_after_timeout(conf_id, int(remaining)))

    def _save_pending_sells(self):
        """미처리 익절 확인 파일 저장."""
        try:
            with open(_PENDING_SELLS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._pending_sell_confirmations, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save pending sells: {e}")

    # --- 알림 발송 ---

    async def send_alert(self, level: str, message: str) -> bool:
        """알림 발송. 4096자 초과 시 자동 분할. 중요 알림은 최대 3회 재시도."""
        if not self.enabled or not self.chat_id:
            return False

        if level not in self.alert_levels and level != "circuit_breaker":
            return False

        critical_levels = {"circuit_breaker", "risk_exit", "error"}
        max_retries = 3 if level in critical_levels else 1

        # 4096자 초과 시 분할 전송
        chunks = self._split_message(message, max_len=4090)

        for chunk in chunks:
            for attempt in range(max_retries):
                try:
                    await self._app.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                        parse_mode="HTML",
                    )
                    break
                except Exception as e:
                    # HTML 파싱 에러 시 plain text로 폴백
                    if "Can't parse entities" in str(e):
                        logger.warning(f"HTML parse failed, falling back to plain text: {e}")
                        try:
                            await self._app.bot.send_message(
                                chat_id=self.chat_id,
                                text=chunk,
                            )
                        except Exception as e2:
                            logger.error(f"Plain text fallback also failed: {e2}")
                        break
                    logger.error(f"Telegram alert failed (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"Telegram alert PERMANENTLY failed [{level}]: {chunk[:100]}")
                return False

        return True

    @staticmethod
    def _split_message(text: str, max_len: int = 4090) -> list[str]:
        """긴 메시지를 텔레그램 한도에 맞게 분할."""
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            # 줄바꿈 기준으로 자르기
            cut = text.rfind("\n", 0, max_len)
            if cut <= 0:
                cut = max_len
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def send_alert_sync(self, level: str, message: str) -> bool:
        """동기 알림 발송 — 스케줄러 스레드에서도 안전하게 동작."""
        if not self.enabled or not self.chat_id or not self._app:
            logger.info(f"[Alert:{level}] {message[:200]}")
            return False

        try:
            loop = self._loop
            if loop and loop.is_running():
                # 스케줄러 스레드 등 외부 스레드에서 호출 시
                asyncio.run_coroutine_threadsafe(self.send_alert(level, message), loop)
            else:
                # 이벤트 루프가 없거나 미실행 시 (드문 케이스)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.send_alert(level, message))
            return True
        except Exception as e:
            logger.warning(f"[Alert:{level}] send failed: {e}")
            return False

    async def send_trade_alert(self, trade: dict):
        """매매 체결 알림."""
        action = trade.get("action", "")
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        status = trade.get("status", "")
        mode = trade.get("mode", "simulation")
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        msg = (
            f"{emoji} {mode_tag} <b>{action}</b> {trade.get('name', '')} ({trade.get('ticker', '')})\n"
            f"수량: {trade.get('quantity', 0):,}주 × {trade.get('price', 0):,.0f}원\n"
            f"금액: {trade.get('amount', 0):,.0f}원\n"
        )

        if trade.get("pnl") is not None:
            pnl = trade["pnl"]
            pnl_emoji = "📈" if pnl > 0 else "📉"
            msg += f"손익: {pnl_emoji} {pnl:,.0f}원 ({trade.get('pnl_pct', '')})\n"

        msg += f"상태: {status}\n"
        if trade.get("reason"):
            msg += f"사유: {trade['reason']}"

        await self.send_alert("trade_executed", msg)

    async def request_confirmation(self, order: dict) -> str:
        """대규모 주문 확인 요청.

        Returns:
            confirmation_id
        """
        conf_id = f"conf_{datetime.now().strftime('%H%M%S%f')}"

        msg = (
            f"⚠️ <b>주문 확인 요청</b>\n\n"
            f"종목: {order.get('name', '')} ({order.get('ticker', '')})\n"
            f"방향: {order.get('action', '')}\n"
            f"수량: {order.get('quantity', 0):,}주\n"
            f"가격: {order.get('price', 0):,.0f}원\n"
            f"금액: {order.get('amount', 0):,.0f}원\n"
            f"사유: {order.get('reason', '')}\n\n"
            f"승인하시겠습니까?"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ 승인", callback_data=f"approve_{conf_id}"),
                InlineKeyboardButton("❌ 거부", callback_data=f"reject_{conf_id}"),
            ]
        ])

        self._pending_confirmations[conf_id] = {
            "order": order,
            "status": "pending",
            "requested_at": datetime.now().isoformat(),
        }

        if self._app:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode="HTML",
                reply_markup=keyboard,
            )

        return conf_id

    async def request_sell_confirmation(self, order_info: dict) -> str:
        """익절 매도 확인 요청 — 수익 중인 포지션 매도 전 사용자 승인.

        타임아웃(기본 5분) 내 응답 없으면 AI 판단대로 자동 익절.

        Returns:
            confirmation_id
        """
        conf_id = f"sell_{datetime.now().strftime('%H%M%S%f')}_{order_info['ticker']}"
        timeout = self._get_confirmation_timeout()

        self._pending_sell_confirmations[conf_id] = {
            "order_info": order_info,
            "status": "pending",
            "requested_at": datetime.now().isoformat(),
        }
        self._save_pending_sells()

        msg = (
            f"💰 <b>익절 매도 확인</b>\n\n"
            f"종목: {order_info.get('name', '')} ({order_info.get('ticker', '')})\n"
            f"수량: {order_info.get('quantity', 0):,}주\n"
            f"현재가: {order_info.get('price', 0):,.0f}원\n"
            f"평가금: {order_info.get('amount', 0):,.0f}원\n"
            f"수익: {order_info.get('pnl', '')}\n"
            f"사유: {order_info.get('reason', '')}\n\n"
            f"익절하시겠습니까?\n"
            f"<i>{timeout // 60}분 내 응답 없으면 AI 판단대로 자동 익절됩니다.</i>"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("💰 익절 실행", callback_data=f"sell_yes_{conf_id}"),
                InlineKeyboardButton("📈 홀드", callback_data=f"sell_no_{conf_id}"),
            ]
        ])

        if self._app:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode="HTML",
                reply_markup=keyboard,
            )

            # 타임아웃 후 자동 익절 스케줄링
            asyncio.ensure_future(self._auto_sell_after_timeout(conf_id, timeout))

        return conf_id

    def _get_confirmation_timeout(self) -> int:
        """확인 타임아웃(초)."""
        if self.system and hasattr(self.system, "config_manager"):
            return self.system.config_manager.get(
                "telegram.confirmation_timeout_seconds", 300
            )
        return 300

    async def _auto_sell_after_timeout(self, conf_id: str, timeout: int):
        """타임아웃 후 응답 없으면 AI 판단대로 자동 익절 실행."""
        await asyncio.sleep(timeout)

        # 아직 pending 상태면 자동 실행 (pop으로 원자적으로 가져와서 이중실행 방지)
        pending = self._pending_sell_confirmations.get(conf_id)
        if not pending or pending["status"] != "pending":
            return  # 이미 사용자가 처리함 (승인/거부)
        pending = self._pending_sell_confirmations.pop(conf_id, None)
        if not pending:
            return  # 다른 경로에서 이미 처리됨
        self._save_pending_sells()

        order_info = pending["order_info"]
        logger.info(f"Sell confirmation timeout — auto-executing: {order_info['name']}({order_info['ticker']})")

        try:
            signal = order_info.get("signal", {"actions": [order_info.get("action_data", {})]})
            sell_signal = {**signal, "actions": [order_info["action_data"]]}
            ticker = order_info["ticker"]
            # blocking I/O는 run_in_executor로 감싸서 이벤트루프 블로킹 방지
            loop = asyncio.get_event_loop()
            fresh_prices = await loop.run_in_executor(
                None, lambda: self.system.data_pipeline.collect_prices_only([ticker])
            )
            current_prices = fresh_prices if fresh_prices.get(ticker) else {ticker: order_info["price"]}
            results = await loop.run_in_executor(
                None, lambda: self.system.executor.execute_signal(sell_signal, current_prices)
            )
            for result in results:
                await loop.run_in_executor(
                    None, lambda r=result: self.system.risk_manager.process_trade_result(r, suppress_notification=True)
                )

            await self.send_alert(
                "trade_executed",
                f"⏰ <b>자동 익절 실행</b> (응답 대기 {timeout // 60}분 초과)\n"
                f"{order_info['name']}({ticker}) "
                f"{order_info['quantity']:,}주 매도\n"
                f"수익: {order_info.get('pnl', '')}",
            )
        except Exception as e:
            logger.error(f"Auto sell execution failed: {e}")
            await self.send_alert(
                "error",
                f"자동 익절 실행 실패: {order_info['name']}({order_info['ticker']})\n{e}",
            )

    # --- 권한 체크 ---

    def _is_owner(self, update: Update) -> bool:
        """메시지 발신자가 등록된 소유자인지 확인."""
        return str(update.effective_chat.id) == str(self.chat_id)

    async def _check_auth(self, update: Update) -> bool:
        """권한 체크 — 소유자가 아니면 무시."""
        if not self._is_owner(update):
            await update.message.reply_text("⛔ 권한이 없습니다.")
            logger.warning(f"Unauthorized access attempt: chat_id={update.effective_chat.id}")
            return False
        return True

    # --- 명령 핸들러 ---

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        await update.message.reply_text(
            "🤖 AI Trader Bot\n\n"
            "사용 가능한 명령:\n"
            "/status - 시스템 상태\n"
            "/portfolio - 포트폴리오 현황 (매입가·손익)\n"
            "/today - 오늘 매매 내역\n"
            "/trades - 최근 매매 내역 (전체)\n"
            "/orders - 당일 주문 체결 현황 (실시간)\n"
            "/analysis - 마지막 LLM 분석 결과 (now: 즉시 실행)\n"
            "/screen - 종목 스크리닝 결과 (now: 즉시 실행)\n"
            "/backtest - 백테스트 (compare/optimize)\n"
            "/strategy - 현재 전략 + 스크리닝 종목\n"
            "/param - 파라미터 조회/변경 (범위 초과 강제 적용 가능)\n"
            "/pause - 거래 일시 중지\n"
            "/resume - 거래 재개\n"
            "/mode <simulation|live> - 모드 전환\n"
            "/setcapital - 초기자본 설정 (auto/±금액/reset)\n"
            "/reset - 서킷브레이커 리셋\n\n"
            "자연어로 전략 변경도 가능합니다.\n"
            "예: '단타로 전환해', '현금비중 50%로 올려'"
        )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        status = self.system.get_status()
        mode = status.get('mode', 'unknown')
        mode_label = "🔴 실거래" if mode == "live" else "🔵 시뮬레이션"
        paused_str = " (⏸ 일시중지)" if status.get("paused") else ""
        watchlist = status.get('watchlist', [])
        watchlist_str = ", ".join(watchlist[:5]) if watchlist else "없음"
        if len(watchlist) > 5:
            watchlist_str += f" 외 {len(watchlist) - 5}개"

        msg = (
            f"📊 <b>시스템 상태</b>\n"
            f"{'━' * 24}\n\n"
            f"모드: {mode_label}{paused_str}\n"
            f"서킷브레이커: {status.get('circuit_state', 'unknown')}\n\n"
            f"<b>자산 현황</b>\n"
            f"  총자산: {status.get('total_asset', 0):,.0f}원\n"
            f"  현금: {status.get('cash', 0):,.0f}원\n"
            f"  수익률: {status.get('total_pnl_pct', '0%')}\n\n"
        )

        # 보유종목 상세
        summary = self.system.portfolio.get_summary()
        positions = summary.get("positions", {})
        if positions:
            # 전략 유형별 그룹핑
            from collections import defaultdict
            by_type = defaultdict(list)
            for ticker, pos in positions.items():
                st = pos.get("strategy_type", "swing")
                by_type[st].append((ticker, pos))

            msg += f"<b>보유종목 ({len(positions)}개)</b>\n"
            type_labels = {"value": "💎 가치투자", "swing": "🔄 스윙", "daytrading": "⚡ 단타"}
            for st in ["value", "swing", "daytrading"]:
                items = by_type.get(st, [])
                if not items:
                    continue
                msg += f"\n<b>{type_labels.get(st, st)}</b>\n"
                for ticker, pos in items:
                    pnl = pos.get("pnl", 0)
                    pnl_pct = pos.get("pnl_pct", "0%")
                    pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
                    msg += (
                        f"  {pnl_emoji} {pos['name']}({ticker})\n"
                        f"     {pos['quantity']}주 | 현재가 {pos['current_price']:,.0f}원\n"
                        f"     매입 {pos['avg_price']:,.0f}원 → {pnl:+,.0f}원({pnl_pct})\n"
                    )
        else:
            msg += f"<b>보유종목: 없음</b>\n"

        msg += (
            f"\n<b>관심종목</b>\n  {watchlist_str}\n"
            f"\nLLM 비용(오늘): ${status.get('llm_daily_cost', 0):.2f}\n"
        )

        # 오늘 매매 요약
        trades = self.system.portfolio.get_today_trades()
        if trades:
            buy_cnt = sum(1 for t in trades if t["action"] == "BUY")
            sell_cnt = sum(1 for t in trades if t["action"] == "SELL")
            realized = sum(t.get("pnl", 0) or 0 for t in trades)
            msg += f"\n<b>오늘 매매</b>: 매수 {buy_cnt}건 / 매도 {sell_cnt}건"
            if realized:
                msg += f" | 실현 {realized:+,.0f}원"
            msg += "\n"

        await update.message.reply_html(msg)

    async def _cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        trades = self.system.portfolio.get_today_trades()
        if not trades:
            await update.message.reply_text("오늘 매매 내역이 없습니다.")
            return

        msg = "📋 <b>오늘의 매매</b>\n\n"
        for t in trades:
            emoji = "🟢" if t["action"] == "BUY" else "🔴"
            mode_tag = "[SIM]" if t.get("mode", "simulation") == "simulation" else "[LIVE]"
            pnl_str = ""
            if t.get("pnl") is not None:
                pnl = t["pnl"]
                pnl_str = f"  손익: {pnl:+,.0f}원 ({t.get('pnl_pct', '')})"
            msg += (
                f"{emoji} {mode_tag} {t['action']} {t['name']}({t['ticker']})\n"
                f"  {t['quantity']:,}주 × {t['price']:,.0f}원 = {t['amount']:,.0f}원\n"
                f"{pnl_str}\n"
            )

        # 오늘 총 손익
        total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        if total_pnl:
            msg += f"\n<b>오늘 실현손익: {total_pnl:+,.0f}원</b>"

        await update.message.reply_html(msg)

    async def _cmd_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        strategy = self.system.llm_engine.load_strategy()
        if len(strategy) > 2500:
            strategy = strategy[:2500] + "\n\n... (생략)"

        msg = f"📜 현재 전략:\n\n{strategy}"

        # 스크리닝 관심종목 추가
        if self.system.screener:
            result = self.system.screener.get_last_result()
            if result and result.candidates:
                ts = result.timestamp[:16].replace("T", " ")
                msg += f"\n\n{'─' * 24}\n🔍 스크리닝 관심종목 ({ts})\n"
                for i, c in enumerate(result.candidates[:8], 1):
                    name = c.get("name", c["ticker"])
                    score = c.get("composite_score", 0)
                    change = c.get("change_pct", 0)
                    msg += f"  {i}. {name}({c['ticker']}) 점수:{score:.0f} {change:+.1f}%\n"

        await update.message.reply_text(msg)

    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        summary = self.system.portfolio.get_summary()
        mode = self.system.config_manager.get_mode()
        mode_label = "🔵시뮬레이션" if mode == "simulation" else "🔴실거래"

        msg = (
            f"💰 <b>포트폴리오</b> ({mode_label})\n\n"
            f"초기자본: {summary.get('initial_capital', 0):,.0f}원\n"
            f"총자산: {summary['total_asset']:,.0f}원\n"
            f"현금: {summary['cash']:,.0f}원 ({summary['cash_ratio']})\n"
            f"평가금: {summary['invested']:,.0f}원\n"
            f"총손익: {summary['total_pnl']:,.0f}원 ({summary['total_pnl_pct']})\n"
            f"{'─' * 24}\n"
        )

        for ticker, pos in summary.get("positions", {}).items():
            pnl_emoji = "📈" if pos["pnl"] > 0 else "📉" if pos["pnl"] < 0 else "➡️"
            cost_total = pos["quantity"] * pos["avg_price"]
            msg += (
                f"\n{pnl_emoji} <b>{pos['name']}</b>({ticker})\n"
                f"  보유: {pos['quantity']}주\n"
                f"  매입단가: {pos['avg_price']:,.0f}원\n"
                f"  매입금액: {cost_total:,.0f}원\n"
                f"  현재가: {pos['current_price']:,.0f}원\n"
                f"  평가금: {pos['market_value']:,.0f}원\n"
                f"  손익: {pos['pnl']:+,.0f}원 ({pos['pnl_pct']})\n"
            )

        await update.message.reply_html(msg)

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if self.system:
            self.system.pause()
        await update.message.reply_text("⏸️ 거래가 일시 중지되었습니다.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if self.system:
            self.system.resume()
        await update.message.reply_text("▶️ 거래가 재개되었습니다.")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        args = context.args
        if not args or args[0] not in ("simulation", "live"):
            await update.message.reply_text("사용법: /mode simulation 또는 /mode live")
            return

        mode = args[0]
        if mode == "live":
            await update.message.reply_text(
                "⚠️ LIVE 모드로 전환하면 실제 주문이 실행됩니다.\n"
                "확실하면 /mode_confirm_live 를 입력하세요."
            )
            return

        if self.system:
            msg = self.system.switch_mode(mode)
            await update.message.reply_text(msg)
        else:
            await update.message.reply_text("시스템이 초기화되지 않았습니다.")

    async def _cmd_mode_confirm_live(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """LIVE 모드 전환 확인 — 실계좌 동기화 포함."""
        if not await self._check_auth(update):
            return
        if self.system:
            await update.message.reply_text("⏳ 실계좌 동기화 중...")
            msg = self.system.switch_mode("live")
            await update.message.reply_text(
                f"{msg}\n\n이제부터 실제 주문이 실행됩니다.\n"
                f"시뮬레이션으로 돌아가려면: /mode simulation"
            )
        else:
            await update.message.reply_text("시스템이 초기화되지 않았습니다.")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if self.system:
            msg = self.system.circuit_breaker.manual_reset()
            await update.message.reply_text(f"🔄 {msg}")
        else:
            await update.message.reply_text("시스템 미연결")

    async def _cmd_setcapital(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """초기자본 수동 설정. /setcapital <금액> 또는 /setcapital auto."""
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        args = context.args
        portfolio = self.system.portfolio
        old_capital = portfolio.initial_capital

        if not args:
            # 인자 없으면 현재 값 + 사용법 안내
            await update.message.reply_text(
                f"💰 <b>초기자본 설정</b>\n\n"
                f"현재 초기자본: {old_capital:,.0f}원\n"
                f"현재 총자산: {portfolio.total_asset:,.0f}원\n"
                f"투자원금(현금+매입금액): {portfolio.cash + sum(p.quantity * p.avg_price for p in portfolio.positions.values()):,.0f}원\n\n"
                f"사용법:\n"
                f"/setcapital auto — 투자원금 기준 자동 설정\n"
                f"/setcapital 50000000 — 5000만원으로 설정\n"
                f"/setcapital +20000000 — 입금/이체 시 2000만원 추가\n"
                f"/setcapital -5000000 — 출금 시 500만원 차감\n"
                f"/setcapital reset — 총자산 기준 리셋 (수익률 0%부터)",
                parse_mode="HTML",
            )
            return

        arg = args[0].strip().lower()

        if arg == "auto":
            # 투자원금(현금 + 매입금액) 기준
            new_capital = portfolio.cash + sum(
                p.quantity * p.avg_price for p in portfolio.positions.values()
            )
        elif arg == "reset":
            # 현재 총자산 기준 (수익률 0%부터 시작)
            new_capital = portfolio.total_asset
        elif arg.startswith("+") or arg.startswith("-"):
            # +/- 증감: 입금, 이체, 출금 시 초기자본 조정
            try:
                delta = float(arg.replace(",", ""))
                new_capital = old_capital + delta
                if new_capital <= 0:
                    await update.message.reply_text(
                        f"변경 후 초기자본이 {new_capital:,.0f}원이 됩니다. 0 이하는 불가합니다."
                    )
                    return
            except ValueError:
                await update.message.reply_text("잘못된 금액입니다. 예: /setcapital +20000000")
                return
        else:
            try:
                new_capital = float(arg.replace(",", ""))
                if new_capital <= 0:
                    raise ValueError
            except ValueError:
                await update.message.reply_text("잘못된 금액입니다. 숫자 또는 auto/reset을 입력하세요.")
                return

        portfolio.initial_capital = new_capital
        if new_capital > portfolio.high_water_mark:
            portfolio.high_water_mark = new_capital
        portfolio._save_state()
        pnl = portfolio.total_asset - new_capital
        pnl_pct = pnl / new_capital * 100 if new_capital > 0 else 0

        # +/- 변경인 경우 delta 표시 포함
        diff = new_capital - old_capital
        delta_str = f" ({diff:+,.0f})" if diff != 0 else ""

        # 입출금 시 전일 스냅샷 보정 (서킷브레이커 오발동 방지)
        if diff != 0:
            try:
                import sqlite3
                db_path = str(Path(__file__).parent.parent / "data" / "storage" / "trader.db")
                with sqlite3.connect(db_path) as conn:
                    # 최근 스냅샷의 total_asset을 입출금 금액만큼 조정
                    conn.execute(
                        "UPDATE daily_snapshot SET total_asset = total_asset + ? "
                        "WHERE date = (SELECT MAX(date) FROM daily_snapshot)",
                        (diff,),
                    )
                logger.info(f"Daily snapshot adjusted by {diff:+,.0f} for capital change")
            except Exception as e:
                logger.warning(f"Snapshot adjustment failed: {e}")

        await update.message.reply_text(
            f"✅ 초기자본 변경 완료\n\n"
            f"이전: {old_capital:,.0f}원\n"
            f"변경: {new_capital:,.0f}원{delta_str}\n"
            f"총자산: {portfolio.total_asset:,.0f}원\n"
            f"수익률: {pnl:+,.0f}원 ({pnl_pct:+.2f}%)"
        )

    async def _cmd_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """당일 주문 체결 현황 (KIS API 실시간 조회)."""
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        await update.message.reply_text("주문 현황 조회 중...")
        try:
            loop = asyncio.get_event_loop()
            orders = await loop.run_in_executor(
                None, self.system.market_client.get_today_orders
            )
        except Exception as e:
            await update.message.reply_text(f"조회 실패: {e}")
            return

        if not orders:
            await update.message.reply_text("오늘 주문 내역이 없습니다.")
            return

        msg = "📋 <b>당일 주문 현황</b>\n\n"
        for o in orders:
            status = "✅체결" if o["filled"] else f"⏳미체결({o['ccld_qty']}/{o['ord_qty']})"
            side_emoji = "🟢" if o["side"] == "매수" else "🔴"
            time_str = f"{o['order_time'][:2]}:{o['order_time'][2:4]}" if len(o['order_time']) >= 4 else ""
            msg += (
                f"{side_emoji} {o['side']} <b>{o['name']}</b>({o['ticker']})\n"
                f"  주문: {o['ord_qty']}주 × {o['ord_price']:,}원\n"
            )
            if o["ccld_qty"] > 0:
                msg += f"  체결: {o['ccld_qty']}주 × {o['ccld_price']:,}원 = {o['ccld_amount']:,}원\n"
            msg += f"  {status} | {time_str} | #{o['order_no']}\n\n"

        await update.message.reply_html(msg)

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """최근 매매 내역 조회."""
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        limit = 10
        if context.args:
            try:
                limit = min(int(context.args[0]), 30)
            except ValueError:
                pass

        trades = self.system.portfolio.get_trade_history(limit=limit)
        if not trades:
            await update.message.reply_text("매매 내역이 없습니다.")
            return

        msg = f"📋 <b>최근 매매 내역</b> (최대 {limit}건)\n\n"
        for t in trades:
            emoji = "🟢" if t["action"] == "BUY" else "🔴"
            mode_tag = "[SIM]" if t.get("mode", "simulation") == "simulation" else "[LIVE]"
            ts = t["timestamp"][:16].replace("T", " ")
            pnl_str = ""
            if t.get("pnl") is not None:
                pnl_str = f" | 손익: {t['pnl']:+,.0f}원"
            msg += (
                f"{emoji} {mode_tag} {ts}\n"
                f"  {t['action']} {t['name']}({t['ticker']})\n"
                f"  {t['quantity']:,}주 × {t['price']:,.0f}원{pnl_str}\n\n"
            )

        await update.message.reply_html(msg)

    async def _cmd_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """마지막 LLM 분석 결과 조회. /analysis now 로 즉시 분석 실행."""
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        # /analysis now → 즉시 분석 실행
        if context.args and context.args[0].lower() == "now":
            await update.message.reply_text("🔄 LLM 분석 즉시 실행 중... (1~2분 소요)")
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.system.cycle_llm_analysis(suppress_notification=True))
            except Exception as e:
                await update.message.reply_text(f"분석 실행 실패: {e}")
                return

        last = self.system.get_last_analysis()
        if not last:
            await update.message.reply_text(
                "아직 분석 결과가 없습니다.\n"
                "/analysis now 로 즉시 실행할 수 있습니다."
            )
            return

        signal = last["signal"]
        actions = last["actions"]
        ts = last["timestamp"][:16].replace("T", " ")

        mode = self.system.config_manager.get_mode()
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        risk = signal.get("risk_assessment", "?")
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
        from html import escape as _esc
        outlook = _esc(signal.get("market_outlook", "없음"))
        reasoning = _esc(signal.get("reasoning", "없음"))

        msg = (
            f"📊 <b>마지막 LLM 분석</b> {mode_tag}\n"
            f"⏰ {ts}\n"
            f"━━━━━━━━━━━━━━\n"
            f"{risk_emoji} 리스크: {risk}\n"
            f"🔍 시장 전망: {outlook}\n"
        )

        # 보유 종목 현황
        summary = self.system.portfolio.get_summary()
        positions = summary.get("positions", {})
        if positions:
            msg += f"\n💼 <b>보유 종목:</b>\n"
            for ticker, pos in positions.items():
                pnl = pos.get("pnl", 0)
                pnl_pct = pos.get("pnl_pct", "0%")
                pnl_emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
                msg += (
                    f"  {pnl_emoji} {pos['name']}({ticker})"
                    f" {pos['quantity']}주"
                    f" | {pnl:+,.0f}원({pnl_pct})\n"
                )
            msg += f"  총자산: {summary['total_asset']:,.0f}원 ({summary['total_pnl_pct']})\n"

        all_actions = signal.get("actions", [])
        if all_actions:
            msg += "\n📋 <b>종목 판단:</b>\n"
            for a in all_actions:
                action_type = a.get("type", "HOLD")
                name = _esc(a.get("name", a.get("ticker", "")))
                reason = _esc(a.get("reason", ""))
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action_type, "⚪")
                ratio_str = ""
                if a.get("ratio"):
                    ratio_str = f" ({a['ratio']:.0%})"
                msg += f"  {emoji} {action_type} {name}{ratio_str}\n"
                if reason:
                    msg += f"     └ {reason}\n"
        else:
            msg += "\n📋 판단: 전종목 HOLD\n"

        executed = sum(1 for a in actions if a.get("type", "").upper() in ("BUY", "SELL"))
        if executed > 0:
            msg += f"\n⚡ 매매 실행: {executed}건\n"

        if reasoning and reasoning != "없음":
            msg += f"\n💬 <b>판단 근거:</b>\n{reasoning}"

        # 4096자 초과 시 분할 전송
        chunks = self._split_message(msg, max_len=4090)
        for chunk in chunks:
            await update.message.reply_html(chunk)

    async def _cmd_screen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """종목 스크리닝 결과 조회. /screen now 로 즉시 실행."""
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return
        if not self.system.screener:
            await update.message.reply_text("스크리너가 비활성화되어 있습니다.")
            return

        # /screen now → 즉시 스크리닝 실행
        if context.args and context.args[0].lower() == "now":
            await update.message.reply_text("🔍 종목 스크리닝 실행 중... (30초~1분 소요)")
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: self.system.cycle_screening(suppress_notification=True))
            except Exception as e:
                await update.message.reply_text(f"스크리닝 실패: {e}")
                return

        result = self.system.screener.get_last_result()
        if not result or not result.candidates:
            await update.message.reply_text(
                "스크리닝 결과가 없습니다.\n"
                "/screen now 로 즉시 실행할 수 있습니다."
            )
            return

        msg = self.system.notifier.format_screening_msg(result)
        await update.message.reply_html(msg)

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """백테스트 실행.

        /backtest — 현재 전략(swing) 백테스트
        /backtest swing|daytrading — 특정 전략
        /backtest compare — 전략 비교
        /backtest optimize — 현재 전략 파라미터 최적화
        /backtest optimize swing|daytrading — 특정 전략 최적화
        """
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        args = [a.lower() for a in (context.args or [])]
        valid_strategies = ["swing", "daytrading"]

        # 명령 파싱
        if not args:
            mode = "single"
            strategy_name = "swing"
        elif args[0] == "results":
            # 최근 백테스트 결과 조회 (DB에서)
            feedback = self.system.analysis_store.build_backtest_feedback(
                self.system._watchlist, list(self.system.portfolio.positions.keys()),
            )
            if feedback:
                await update.message.reply_text(f"📊 최근 백테스트 결과\n\n{feedback}")
            else:
                await update.message.reply_text(
                    "저장된 백테스트 결과가 없습니다.\n"
                    "/backtest compare 로 실행하세요."
                )
            return
        elif args[0] == "compare":
            mode = "compare"
            strategy_name = ""
        elif args[0] == "optimize":
            mode = "optimize"
            strategy_name = args[1] if len(args) > 1 and args[1] in valid_strategies else "swing"
        elif args[0] in valid_strategies:
            mode = "single"
            strategy_name = args[0]
        else:
            await update.message.reply_text(
                "사용법:\n"
                "/backtest — 스윙 전략 백테스트\n"
                "/backtest swing|daytrading\n"
                "/backtest compare — 전략 비교\n"
                "/backtest optimize [전략] — 최적화\n"
                "/backtest results — 최근 백테스트 결과 조회"
            )
            return

        time_msg = "1~2분" if mode != "optimize" else "3~5분"
        label = {
            "single": f"{strategy_name} 백테스트",
            "compare": "3개 전략 비교",
            "optimize": f"{strategy_name} 최적화",
        }[mode]
        await update.message.reply_text(f"🔬 {label} 실행 중... ({time_msg} 소요)")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.system.run_backtest(mode=mode, strategy_name=strategy_name),
            )
            # 결과가 여러 메시지일 수 있음 (길면 분할)
            for msg in result:
                await update.message.reply_html(msg)
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            await update.message.reply_text(f"백테스트 실패: {e}")

    async def _cmd_param(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """트레이딩 파라미터 직접 조회/변경.

        /param — 현재 파라미터 목록
        /param <key> <value> — 파라미터 변경 (범위 초과 시 확인 요청)
        """
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        args = context.args or []

        # /param → 현재 파라미터 목록
        if not args:
            params = self.system.config_manager.get_all_params_flat()
            limits = self.system.config_manager.safety_rules.get("adjustable_limits", {})
            msg = "⚙️ <b>트레이딩 파라미터</b>\n\n"
            for key, val in params.items():
                limit_key = key.replace(".", "_")
                limit = limits.get(limit_key)
                if limit:
                    range_str = f" [{limit['min']}~{limit['max']}]"
                else:
                    range_str = " [제한없음]"
                msg += f"<code>{key}</code>: {val}{range_str}\n"
            msg += (
                f"\n{'─' * 24}\n"
                "변경: /param &lt;키&gt; &lt;값&gt;\n"
                "예: /param take_profit_pct 0.20\n"
                "범위 초과 시 확인 후 강제 적용 가능"
            )
            await update.message.reply_html(msg)
            return

        # /param <key> <value>
        if len(args) < 2:
            await update.message.reply_text(
                "사용법: /param <키> <값>\n"
                "예: /param take_profit_pct 0.20\n"
                "목록 보기: /param"
            )
            return

        param_key = args[0]
        try:
            value = float(args[1])
            # 정수로 표현 가능하면 정수로
            if value == int(value) and "." not in args[1]:
                value = int(value)
        except ValueError:
            await update.message.reply_text(f"숫자 값이 필요합니다: {args[1]}")
            return

        # 현재값 확인
        current = self.system.config_manager.get_all_params_flat()
        if param_key not in current:
            await update.message.reply_text(
                f"존재하지 않는 파라미터: {param_key}\n"
                "/param 으로 목록을 확인하세요."
            )
            return

        old_value = current[param_key]

        # 검증
        in_whitelist, in_range, msg = self.system.config_manager.validate_and_describe(param_key, value)

        if in_whitelist and in_range:
            # 정상 범위 내 → 즉시 적용
            ok, result_msg, _ = self.system.config_manager.force_set_param(param_key, value)
            if ok:
                await update.message.reply_html(
                    f"✅ <b>파라미터 변경 완료</b>\n\n"
                    f"<code>{param_key}</code>: {old_value} → {value}"
                )
            else:
                await update.message.reply_text(f"변경 실패: {result_msg}")
            return

        # 범위 초과 또는 비허용 → 경고 + 확인 버튼
        change_id = f"fp_{datetime.now().strftime('%H%M%S')}_{param_key.replace('.', '_')}"
        self._pending_param_changes[change_id] = {
            "param": param_key,
            "value": value,
            "old_value": old_value,
        }

        if not in_whitelist:
            warning = f"⚠️ LLM 조정 허용 목록에 없는 파라미터입니다."
        else:
            warning = f"⚠️ 안전 범위 초과: {msg}"

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔓 강제 적용", callback_data=f"force_param_{change_id}"),
                InlineKeyboardButton("❌ 취소", callback_data=f"cancel_param_{change_id}"),
            ]
        ])

        await update.message.reply_html(
            f"⚠️ <b>범위 초과 파라미터 변경</b>\n\n"
            f"<code>{param_key}</code>: {old_value} → {value}\n\n"
            f"{warning}\n\n"
            f"강제 적용하시겠습니까?",
            reply_markup=keyboard,
        )

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """인라인 버튼 콜백 처리 (주문 승인/거부 + 파라미터 강제 적용)."""
        query = update.callback_query
        await query.answer()

        data = query.data

        # 파라미터 강제 적용
        if data.startswith("force_param_"):
            change_id = data.replace("force_param_", "")
            pending = self._pending_param_changes.pop(change_id, None)
            if not pending:
                await query.edit_message_text("⏰ 요청이 만료되었습니다.")
                return
            ok, result_msg, _ = self.system.config_manager.force_set_param(
                pending["param"], pending["value"]
            )
            if ok:
                await query.edit_message_text(
                    f"🔓 강제 적용 완료\n{pending['param']}: {pending['old_value']} → {pending['value']}"
                )
            else:
                await query.edit_message_text(f"변경 실패: {result_msg}")
            return

        if data.startswith("cancel_param_"):
            change_id = data.replace("cancel_param_", "")
            self._pending_param_changes.pop(change_id, None)
            await query.edit_message_text("❌ 파라미터 변경이 취소되었습니다.")
            return

        # 익절 매도 승인/거부
        if data.startswith("sell_yes_"):
            conf_id = data.replace("sell_yes_", "")
            pending = self._pending_sell_confirmations.pop(conf_id, None)
            self._save_pending_sells()
            if not pending:
                await query.edit_message_text("⏰ 요청이 만료되었습니다.")
                return
            order_info = pending["order_info"]
            try:
                signal = order_info.get("signal", {"actions": [order_info.get("action_data", {})]})
                sell_signal = {**signal, "actions": [order_info["action_data"]]}
                ticker = order_info["ticker"]
                loop = asyncio.get_event_loop()
                fresh_prices = await loop.run_in_executor(
                    None, lambda: self.system.data_pipeline.collect_prices_only([ticker])
                )
                current_prices = fresh_prices if fresh_prices.get(ticker) else {ticker: order_info["price"]}
                results = await loop.run_in_executor(
                    None, lambda: self.system.executor.execute_signal(sell_signal, current_prices)
                )
                for result in results:
                    await loop.run_in_executor(
                        None, lambda r=result: self.system.risk_manager.process_trade_result(r, suppress_notification=True)
                    )
                await query.edit_message_text(
                    f"💰 익절 실행 완료\n"
                    f"{order_info['name']}({ticker}) "
                    f"{order_info['quantity']:,}주 매도\n"
                    f"수익: {order_info.get('pnl', '')}"
                )
            except Exception as e:
                await query.edit_message_text(f"매도 실행 실패: {e}")
            return

        if data.startswith("sell_no_"):
            conf_id = data.replace("sell_no_", "")
            self._pending_sell_confirmations.pop(conf_id, None)
            self._save_pending_sells()
            await query.edit_message_text("📈 홀드 — 익절을 보류합니다.")
            return

        # optimizer 제안 적용/거부
        if data == "opt_apply":
            try:
                import yaml
                pending_path = Path(__file__).parent.parent / "data" / "storage" / "pending_optimizer_suggestion.json"
                if not pending_path.exists():
                    await query.edit_message_text("⏰ 제안이 만료되었습니다.")
                    return
                with open(pending_path) as f:
                    suggestion = json.load(f)
                params = suggestion.get("suggested_params", {})
                # trading-params.yaml 업데이트
                tp_path = Path(__file__).parent.parent / "config" / "trading-params.yaml"
                with open(tp_path) as f:
                    tp = yaml.safe_load(f)
                if "take_profit_pct" in params:
                    tp["take_profit_pct"] = params["take_profit_pct"]
                if "stop_loss_pct" in params:
                    tp["stop_loss_pct"] = -abs(params["stop_loss_pct"])
                if "position_size_pct" in params:
                    tp["position_size_pct"] = params["position_size_pct"]
                if "max_hold_days" in params:
                    tp.setdefault("holding_period_days", {})["max"] = params["max_hold_days"]
                if "rsi_oversold" in params:
                    tp.setdefault("indicators", {})["rsi_oversold"] = params["rsi_oversold"]
                with open(tp_path, "w") as f:
                    yaml.dump(tp, f, default_flow_style=False, allow_unicode=True)
                pending_path.unlink()
                await query.edit_message_text(
                    f"✅ 파라미터가 적용되었습니다!\n\n"
                    f"다음 분석 사이클부터 반영됩니다.\n"
                    f"즉시 반영하려면 PM2 재시작이 필요합니다."
                )
                logger.info(f"Optimizer suggestion applied: {params}")
            except Exception as e:
                await query.edit_message_text(f"❌ 적용 실패: {e}")
            return

        if data == "opt_reject":
            pending_path = Path(__file__).parent.parent / "data" / "storage" / "pending_optimizer_suggestion.json"
            if pending_path.exists():
                pending_path.unlink()
            await query.edit_message_text("❌ 제안을 무시했습니다. 현재 전략을 유지합니다.")
            return

        # 기존: 주문 승인/거부
        if data.startswith("approve_"):
            conf_id = data.replace("approve_", "")
            conf = self._pending_confirmations.pop(conf_id, None)
            if conf and conf["status"] == "pending" and self.system:
                conf["status"] = "approved"
                order = conf["order"]
                try:
                    signal = {"actions": [order]}
                    ticker = order.get("ticker", "")
                    prices = self.system.data_pipeline.collect_prices_only([ticker])
                    results = self.system.executor.execute_signal(signal, prices)
                    for result in results:
                        self.system.risk_manager.process_trade_result(result)
                    await query.edit_message_text(
                        f"✅ 주문 승인 — 실행 완료\n"
                        f"{order.get('name', '')} {order.get('action', '')} "
                        f"{order.get('quantity', 0):,}주"
                    )
                except Exception as e:
                    logger.error(f"Approved order execution failed: {e}")
                    await query.edit_message_text(f"✅ 승인했으나 실행 실패: {e}")
            elif not conf:
                await query.edit_message_text("⏰ 만료된 주문입니다.")
        elif data.startswith("reject_"):
            conf_id = data.replace("reject_", "")
            conf = self._pending_confirmations.pop(conf_id, None)
            if conf:
                await query.edit_message_text("❌ 주문이 거부되었습니다.")
            else:
                await query.edit_message_text("⏰ 만료된 주문입니다.")

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """자연어 메시지 처리 → 전략 변경."""
        if not self._is_owner(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        user_text = update.message.text.strip()
        if not user_text:
            return

        await update.message.reply_text("🤔 분석 중...")

        try:
            result = self.system.handle_natural_language(user_text)
            msg = f"📝 <b>해석 결과</b>\n\n"

            if result.get("interpretation"):
                msg += f"{result['interpretation']}\n\n"

            adjustments = result.get("adjustments", [])
            if adjustments:
                msg += "<b>파라미터 조정:</b>\n"
                for adj in adjustments:
                    status = "✅" if adj.get("applied", False) else "❌"
                    msg += f"{status} {adj['param']}: {adj.get('old_value', '?')} → {adj['value']}\n"

            if result.get("strategy_change_needed"):
                msg += f"\n전략 변경 제안: {result.get('strategy_suggestion', '')}"

            await update.message.reply_html(msg)

        except Exception as e:
            logger.error(f"Natural language processing failed: {e}")
            await update.message.reply_text(f"처리 실패: {e}")

    def get_confirmation_status(self, conf_id: str) -> str:
        conf = self._pending_confirmations.get(conf_id, {})
        return conf.get("status", "unknown")
