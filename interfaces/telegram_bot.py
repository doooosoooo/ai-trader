"""Telegram 봇 — 양방향 알림 + 명령 수신."""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Callable

from loguru import logger

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
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram bot started")

    async def stop(self):
        if self._app:
            try:
                if self._app.updater and self._app.updater.running:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception:
                pass

    # --- 알림 발송 ---

    async def send_alert(self, level: str, message: str) -> bool:
        """알림 발송."""
        if not self.enabled or not self.chat_id:
            return False

        if level not in self.alert_levels and level != "circuit_breaker":
            return False

        try:
            await self._app.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="HTML",
            )
            return True
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
            return False

    def send_alert_sync(self, level: str, message: str) -> bool:
        """동기 알림 발송 (콜백용)."""
        if not self.enabled or not self.chat_id or not self._app:
            logger.info(f"[Alert:{level}] {message}")
            return False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_alert(level, message))
            else:
                loop.run_until_complete(self.send_alert(level, message))
            return True
        except Exception:
            logger.info(f"[Alert:{level}] {message}")
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
        conf_id = f"conf_{datetime.now().strftime('%H%M%S')}"

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
            "/strategy - 현재 전략\n"
            "/pause - 거래 일시 중지\n"
            "/resume - 거래 재개\n"
            "/mode <simulation|live> - 모드 전환\n"
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
        watchlist = status.get('watchlist', [])
        watchlist_str = ", ".join(watchlist) if watchlist else "없음"
        msg = (
            f"📊 <b>시스템 상태</b>\n\n"
            f"모드: {status.get('mode', 'unknown')}\n"
            f"서킷브레이커: {status.get('circuit_state', 'unknown')}\n"
            f"총자산: {status.get('total_asset', 0):,.0f}원\n"
            f"수익률: {status.get('total_pnl_pct', '0%')}\n"
            f"보유종목: {status.get('num_positions', 0)}개\n"
            f"관심종목: {watchlist_str}\n"
            f"LLM 일일비용: ${status.get('llm_daily_cost', 0):.2f}\n"
        )
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
        # 너무 길면 잘라서 보냄
        if len(strategy) > 3000:
            strategy = strategy[:3000] + "\n\n... (생략)"
        await update.message.reply_text(f"📜 현재 전략:\n\n{strategy}")

    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        summary = self.system.portfolio.get_summary()
        mode = self.system.config_manager.settings.get("mode", "simulation")
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
            self.system.config_manager.set_mode(mode)
        await update.message.reply_text(f"모드 전환: {mode}")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if self.system:
            msg = self.system.circuit_breaker.manual_reset()
            await update.message.reply_text(f"🔄 {msg}")
        else:
            await update.message.reply_text("시스템 미연결")

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

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """인라인 버튼 콜백 처리 (주문 승인/거부)."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data.startswith("approve_"):
            conf_id = data.replace("approve_", "")
            if conf_id in self._pending_confirmations:
                self._pending_confirmations[conf_id]["status"] = "approved"
                await query.edit_message_text("✅ 주문이 승인되었습니다.")
        elif data.startswith("reject_"):
            conf_id = data.replace("reject_", "")
            if conf_id in self._pending_confirmations:
                self._pending_confirmations[conf_id]["status"] = "rejected"
                await query.edit_message_text("❌ 주문이 거부되었습니다.")

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
