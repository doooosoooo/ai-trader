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
        self._app.add_handler(CommandHandler("analysis", self._cmd_analysis))
        self._app.add_handler(CommandHandler("screen", self._cmd_screen))
        self._app.add_handler(CommandHandler("backtest", self._cmd_backtest))
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
            "/analysis - 마지막 LLM 분석 결과 (now: 즉시 실행)\n"
            "/screen - 종목 스크리닝 결과 (now: 즉시 실행)\n"
            "/backtest - 백테스트 (compare/optimize)\n"
            "/strategy - 현재 전략 + 스크리닝 종목\n"
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
                await loop.run_in_executor(None, self.system.cycle_llm_analysis)
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
        outlook = signal.get("market_outlook", "없음")
        reasoning = signal.get("reasoning", "없음")

        msg = (
            f"📊 <b>마지막 LLM 분석</b> {mode_tag}\n"
            f"⏰ {ts}\n"
            f"━━━━━━━━━━━━━━\n"
            f"{risk_emoji} 리스크: {risk}\n"
            f"🔍 시장 전망: {outlook}\n"
        )

        all_actions = signal.get("actions", [])
        if all_actions:
            msg += "\n📋 <b>종목 판단:</b>\n"
            for a in all_actions:
                action_type = a.get("type", "HOLD")
                name = a.get("name", a.get("ticker", ""))
                reason = a.get("reason", "")
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
            if len(reasoning) > 300:
                reasoning = reasoning[:300] + "..."
            msg += f"\n💬 <b>판단 근거:</b>\n{reasoning}"

        await update.message.reply_html(msg)

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
                await loop.run_in_executor(None, self.system.cycle_screening)
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

        msg = self.system._format_screening_msg(result)
        await update.message.reply_html(msg)

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """백테스트 실행.

        /backtest — 현재 전략(swing) 백테스트
        /backtest swing|daytrading|defensive — 특정 전략
        /backtest compare — 3개 전략 비교
        /backtest optimize — 현재 전략 파라미터 최적화
        /backtest optimize swing|daytrading|defensive — 특정 전략 최적화
        """
        if not await self._check_auth(update):
            return
        if not self.system:
            await update.message.reply_text("시스템 미연결")
            return

        args = [a.lower() for a in (context.args or [])]
        valid_strategies = ["swing", "daytrading", "defensive"]

        # 명령 파싱
        if not args:
            mode = "single"
            strategy_name = "swing"
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
                "/backtest swing|daytrading|defensive\n"
                "/backtest compare — 3개 전략 비교\n"
                "/backtest optimize [전략] — 최적화"
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
