"""텔레그램 메시지 포맷 + 알림 서비스."""

import asyncio
import html
from datetime import datetime

from loguru import logger

from core.analysis_store import ticker_display


class NotificationService:
    """메시지 포맷팅 및 텔레그램 알림 전송을 담당."""

    def __init__(self, telegram, portfolio, config_manager, event_loop_fn=None):
        """
        Args:
            telegram: TelegramBot 인스턴스
            portfolio: Portfolio 인스턴스
            config_manager: ConfigManager 인스턴스
            event_loop_fn: asyncio event loop 반환 콜백 (request_sell_confirmation용)
        """
        self.telegram = telegram
        self.portfolio = portfolio
        self.config_manager = config_manager
        self._get_event_loop = event_loop_fn

    def notify(self, level: str = "", message: str = ""):
        """Telegram 알림 (동기 래퍼)."""
        self.telegram.send_alert_sync(level, message)

    def format_analysis_msg(self, signal: dict, actions: list) -> str:
        """LLM 분석 결과를 텔레그램 메시지로 포맷."""
        now = datetime.now().strftime("%H:%M")
        mode = self.config_manager.get_mode()
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        risk = signal.get("risk_assessment", "?")
        risk_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
        outlook = html.escape(signal.get("market_outlook", "없음"))
        reasoning = html.escape(signal.get("reasoning", "없음"))

        msg = (
            f"📊 LLM 분석 결과 {mode_tag} [{now}]\n"
            f"━━━━━━━━━━━━━━\n"
            f"{risk_emoji} 리스크: {risk}\n"
            f"🔍 시장 전망: {outlook}\n"
        )

        # 종목별 판단
        all_actions = signal.get("actions", [])
        if all_actions:
            msg += "\n📋 종목 판단:\n"
            for a in all_actions:
                action_type = a.get("type", "HOLD")
                ticker = a.get("ticker", "")
                name = html.escape(a.get("name", ticker_display(ticker) if ticker else ""))
                reason = html.escape(a.get("reason", ""))
                emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action_type, "⚪")
                ratio_str = ""
                if a.get("ratio"):
                    ratio_str = f" ({a['ratio']:.0%})"
                msg += f"  {emoji} {action_type} {name}{ratio_str}"
                if reason:
                    msg += f"\n     └ {reason}"
                msg += "\n"
        else:
            msg += "\n📋 판단: 전종목 HOLD\n"

        # 실행된 매매 수 (HOLD 제외)
        executed = sum(1 for a in actions if a.get("type", "").upper() in ("BUY", "SELL"))
        if executed > 0:
            msg += f"\n⚡ 매매 실행: {executed}건"

        # 판단 근거 (4096자 초과 시 send_alert에서 자동 분할)
        if reasoning and reasoning != "없음":
            msg += f"\n\n💬 근거:\n{reasoning}"

        return msg

    @staticmethod
    def format_trade_msg(trade: dict) -> str:
        """매매 결과를 텔레그램 메시지로 포맷."""
        action = trade.get("action", "")
        emoji = "🟢" if action == "BUY" else "🔴"
        name = trade.get("name", trade.get("ticker", ""))
        ticker = trade.get("ticker", "")
        msg = (
            f"{emoji} {action} {name}({ticker})\n"
            f"수량: {trade.get('quantity', 0):,}주 @ {trade.get('price', 0):,.0f}원\n"
            f"금액: {trade.get('amount', 0):,.0f}원"
        )
        if trade.get("pnl") is not None:
            pnl = trade["pnl"]
            pnl_emoji = "📈" if pnl > 0 else "📉"
            msg += f"\n손익: {pnl_emoji} {pnl:,.0f}원 ({trade.get('pnl_pct', '')})"
        msg += f"\n상태: {trade.get('status', '')}"
        if trade.get("reason"):
            msg += f"\n사유: {trade['reason']}"
        return msg

    def format_screening_msg(self, result) -> str:
        """스크리닝 결과 텔레그램 메시지 포맷."""
        ts = result.timestamp[:16].replace("T", " ")
        stats = result.screening_stats
        mode = self.config_manager.get_mode()
        mode_tag = "[SIM]" if mode == "simulation" else "[LIVE]"

        msg = (
            f"🔍 <b>종목 스크리닝 결과</b> {mode_tag}\n"
            f"⏰ {ts}\n"
            f"━━━━━━━━━━━━━━\n"
            f"분석: {stats.get('total_analyzed', 0):,}종목 → "
            f"필터: {stats.get('after_filter', 0)}종목 → "
            f"선정: {len(result.candidates)}종목\n\n"
        )

        for i, c in enumerate(result.candidates[:8], 1):
            name = c.get("name", c["ticker"])
            score = c.get("composite_score", 0)
            change = c.get("change_pct", 0)
            ch_emoji = "📈" if change > 0 else ("📉" if change < 0 else "➡️")
            msg += (
                f"{i}. <b>{name}</b>({c['ticker']}) "
                f"점수: {score:.0f} {ch_emoji}{change:+.1f}%\n"
                f"   M:{c.get('momentum_score', 0):.0f} "
                f"V:{c.get('value_score', 0):.0f} "
                f"거:{c.get('volume_score', 0):.0f} "
                f"수:{c.get('flow_score', 0):.0f} "
                f"기:{c.get('technical_score', 0):.0f}\n"
            )

        if result.held_tickers_added:
            held_names = [ticker_display(t) for t in result.held_tickers_added]
            msg += f"\n📌 보유종목 추가: {', '.join(held_names)}"

        # 시장 요약
        kospi = result.market_summary.get("kospi", {})
        if kospi:
            msg += (
                f"\n\n📊 KOSPI: 상승 {kospi.get('advancing', 0)} / "
                f"하락 {kospi.get('declining', 0)} "
                f"(평균 {kospi.get('avg_change_pct', 0):+.2f}%)"
            )

        return msg

    @staticmethod
    def format_backtest_summary(results: list[dict], start: str, end: str) -> str:
        """백테스트 요약 텔레그램 메시지."""
        msg = (
            f"📊 <b>주간 백테스트 결과</b>\n"
            f"기간: {start[:4]}-{start[4:6]}-{start[6:]} ~ {end[:4]}-{end[4:6]}-{end[6:]}\n"
            f"━━━━━━━━━━━━━━\n"
        )
        for r in results:
            ret_emoji = "🟢" if r["return"] > 0 else "🔴"
            opt_tag = " ⚙️" if r.get("optimized") else ""
            msg += (
                f"\n{ret_emoji} <b>{r['name']}</b>{opt_tag}\n"
                f"  수익률: {r['return']:+.1%} | 승률: {r['win_rate']:.0%}\n"
                f"  샤프: {r['sharpe']:.2f} | MDD: {r['mdd']:.1%}\n"
                f"  거래: {r['trades']}건\n"
            )
        # 최고 전략 추천
        if results:
            best = max(results, key=lambda x: x["sharpe"])
            msg += f"\n💡 최적 전략: <b>{best['name']}</b> (샤프 {best['sharpe']:.2f})"
        return msg

    def send_daily_report(self):
        """장 마감 일일 손익 리포트 생성 및 발송."""
        summary = self.portfolio.get_summary()
        trades = self.portfolio.get_today_trades()
        mode = self.config_manager.get_mode()
        mode_label = "🔵시뮬레이션" if mode == "simulation" else "🔴실거래"

        # 오늘 실현 손익
        realized_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        buy_count = sum(1 for t in trades if t["action"] == "BUY")
        sell_count = sum(1 for t in trades if t["action"] == "SELL")

        # 보유 종목 평가 손익
        unrealized_pnl = sum(
            p.pnl for p in self.portfolio.positions.values()
        )

        msg = (
            f"📊 <b>일일 리포트</b> ({mode_label})\n"
            f"{'─' * 26}\n\n"
            f"<b>자산 현황</b>\n"
            f"  총자산: {summary['total_asset']:,.0f}원\n"
            f"  현금: {summary['cash']:,.0f}원 ({summary['cash_ratio']})\n"
            f"  평가금: {summary['invested']:,.0f}원\n\n"
            f"<b>오늘 매매</b>\n"
            f"  매수 {buy_count}건 / 매도 {sell_count}건\n"
            f"  실현손익: {realized_pnl:+,.0f}원\n\n"
            f"<b>보유 종목 평가</b>\n"
        )

        for pos in self.portfolio.positions.values():
            pnl_emoji = "📈" if pos.pnl > 0 else "📉" if pos.pnl < 0 else "➡️"
            msg += (
                f"  {pnl_emoji} {pos.name}({pos.ticker})\n"
                f"    {pos.quantity}주 | 매입 {pos.avg_price:,.0f} → 현재 {pos.current_price:,.0f}\n"
                f"    평가손익: {pos.pnl:+,.0f}원 ({pos.pnl_pct:+.2%})\n"
            )

        msg += (
            f"\n{'─' * 26}\n"
            f"<b>미실현손익: {unrealized_pnl:+,.0f}원</b>\n"
            f"<b>총손익(누적): {summary['total_pnl']:,.0f}원 ({summary['total_pnl_pct']})</b>"
        )

        self.telegram.send_alert_sync("daily_report", msg)

    def request_sell_confirmation(
        self, action: dict, signal: dict, current_price: float,
        pnl: float, pnl_pct: float,
    ) -> None:
        """익절 매도 확인 요청을 텔레그램으로 전송."""
        ticker = action["ticker"]
        name = action.get("name", ticker)
        pos = self.portfolio.positions.get(ticker)
        if not pos:
            return

        order_info = {
            "ticker": ticker,
            "name": name,
            "action": "SELL",
            "quantity": int(pos.quantity * action.get("ratio", 1.0)) or pos.quantity,
            "price": current_price,
            "amount": int((int(pos.quantity * action.get("ratio", 1.0)) or pos.quantity) * current_price),
            "reason": action.get("reason", ""),
            "pnl": f"{pnl:+,.0f}원 ({pnl_pct:+.1f}%)",
            "signal": signal,
            "action_data": action,
        }

        # 텔레그램 확인 요청 (스레드에서도 동작하도록 run_coroutine_threadsafe 사용)
        try:
            loop = self._get_event_loop() if self._get_event_loop else None
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.telegram.request_sell_confirmation(order_info),
                    loop,
                )
            else:
                asyncio.run(self.telegram.request_sell_confirmation(order_info))
            logger.info(f"Sell confirmation requested: {name}({ticker}) PnL: {pnl:+,.0f}원")
        except Exception as e:
            logger.warning(f"Failed to request sell confirmation: {e}")
            # 확인 실패 시 자동 실행하지 않음 (안전)
