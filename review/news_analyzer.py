"""News Deep Analysis — 4시간마다 Opus 4.7로 누적 뉴스를 심층 분석.

스케줄: 평일 08/11/13/15 KST 4회 (scheduler.py news_deep_analysis cron)
결과 활용:
1. data/storage/news_deep_latest.json — 다음 trading 사이클의 news_summary 자리에 주입
2. logs/reviews/news_deep_YYYYMMDD_HHMM.{json,md} — 사람 검수용
3. 텔레그램 news_deep 레벨 알림

가드레일:
- 누적 뉴스 < 10건이면 skip (시간/비용 낭비)
- 동시 실행 방지 (threading.Lock)
- 비동기 threading.Thread(daemon=True) — 사이클 블로킹 방지
- LLM 비용 한도 자동 차단 ($5/일, llm_engine category='news')
- 최신 캐시 TTL: 6시간 (다음 발사 전 만료 안 됨)
"""

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
NEWS_HISTORY_PATH = PROJECT_ROOT / "data" / "news_history.jsonl"
LATEST_CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "news_deep_latest.json"
REVIEWS_DIR = PROJECT_ROOT / "logs" / "reviews"

REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

_MIN_NEWS_FOR_ANALYSIS = 10        # 누적이 이 미만이면 skip
_LOOKBACK_HOURS = 4                # 분석 대상 윈도우
_CACHE_TTL_HOURS = 6               # latest 캐시가 trading LLM에 주입되는 최대 나이


class NewsAnalyzer:
    """뉴스 심층 분석 — Opus 4.7."""

    def __init__(self, llm_engine, portfolio, market_client, watchlist_provider, telegram):
        self.llm = llm_engine
        self.portfolio = portfolio
        self.market_client = market_client
        # watchlist_provider: callable → list[str] (main._watchlist 동적 조회)
        self.watchlist_provider = watchlist_provider
        self.telegram = telegram

        self._lock = threading.Lock()
        self._running = False  # 동시 실행 방지

    # ============================================================
    # 메인 진입점
    # ============================================================

    def trigger(self) -> bool:
        """심층 분석 트리거 — 동시실행 방지 + 누적 뉴스 충분성 체크 후 백그라운드 스레드 fire.

        반환: 실제 실행 여부.
        """
        with self._lock:
            if self._running:
                logger.info("News deep analysis already running — skip")
                return False
            self._running = True

        try:
            news_items = self._load_recent_news(hours=_LOOKBACK_HOURS)
        except Exception as e:
            logger.warning(f"News history load failed: {e}")
            with self._lock:
                self._running = False
            return False

        if len(news_items) < _MIN_NEWS_FOR_ANALYSIS:
            logger.info(
                f"News deep skip — only {len(news_items)} items in last {_LOOKBACK_HOURS}h "
                f"(min={_MIN_NEWS_FOR_ANALYSIS})"
            )
            with self._lock:
                self._running = False
            return False

        t = threading.Thread(
            target=self._run_analysis,
            args=(news_items,),
            daemon=True,
            name="news-deep",
        )
        t.start()
        logger.info(f"News deep analysis triggered: {len(news_items)} items")
        return True

    def _run_analysis(self, news_items: list[dict]) -> None:
        """백그라운드 스레드에서 LLM 호출 + 저장 + 알림."""
        try:
            watchlist_meta = self._build_watchlist_meta()
            holdings_meta = self._build_holdings_meta()
            market_context = self._build_market_context()
            previous = self._load_previous_implications()

            result = self.llm.generate_news_deep_analysis(
                news_items=news_items,
                watchlist_meta=watchlist_meta,
                holdings_meta=holdings_meta,
                market_context=market_context,
                previous_implications=previous,
            )

            if not result:
                logger.error("News deep analysis returned empty")
                return

            result["timestamp"] = datetime.now().isoformat()
            result["news_count"] = len(news_items)
            result["lookback_hours"] = _LOOKBACK_HOURS

            self._save_latest(result)
            self._save_report_files(result)
            self._send_telegram(result)
            logger.info(
                f"News deep complete: {len(result.get('sector_trends', []))} sectors, "
                f"{len(result.get('ticker_impacts', []))} ticker impacts, "
                f"{len(result.get('risk_events', []))} risks"
            )

        except Exception as e:
            logger.error(f"News deep analysis failed: {e}")
            self._notify_failure(str(e))
        finally:
            with self._lock:
                self._running = False

    # ============================================================
    # 외부 인터페이스 — trading LLM 사이클이 호출
    # ============================================================

    def get_latest_deep_summary(self, max_age_hours: float = _CACHE_TTL_HOURS) -> str:
        """다음 trading 사이클의 news_summary 자리에 주입할 텍스트.

        캐시 없거나 만료되면 빈 문자열 반환 (호출자가 fallback으로 헤드라인 요약 사용).
        """
        cached = self._load_latest()
        if not cached:
            return ""

        ts_str = cached.get("timestamp", "")
        if not ts_str:
            return ""
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            return ""

        age = datetime.now() - ts
        if age > timedelta(hours=max_age_hours):
            logger.debug(f"News deep cache expired ({age.total_seconds()/3600:.1f}h old) — fallback to headlines")
            return ""

        implications = cached.get("trading_implications", "") or ""
        if not implications:
            return ""

        # trading LLM이 읽기 쉽게 헤더 + 본문 + 핵심 리스크만 결합
        lines = [
            f"### Opus 뉴스 심층 분석 ({ts.strftime('%m-%d %H:%M')} 기준, {cached.get('news_count', 0)}건 누적)",
            implications,
        ]

        risk_events = cached.get("risk_events", []) or []
        high_risks = [r for r in risk_events if (r.get("severity") or "").lower() == "high"]
        if high_risks:
            lines.append("")
            lines.append("**🔴 High 리스크 이벤트:**")
            for r in high_risks[:3]:
                lines.append(f"- {r.get('event', '')[:200]} (영향: {', '.join(r.get('affected_sectors', []) or [])[:80]})")

        return "\n".join(lines)

    # ============================================================
    # 데이터 수집
    # ============================================================

    def _load_recent_news(self, hours: int) -> list[dict]:
        """news_history.jsonl에서 최근 N시간 뉴스 추출 + 중복 제거."""
        if not NEWS_HISTORY_PATH.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        seen_titles: set[str] = set()
        items: list[dict] = []

        try:
            with open(NEWS_HISTORY_PATH, "r", encoding="utf-8") as f:
                # tail 읽기 — 파일이 클 수 있으니 마지막 200줄만
                lines = f.readlines()[-200:]
        except Exception as e:
            logger.warning(f"news_history.jsonl read failed: {e}")
            return []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("timestamp", "") < cutoff_iso:
                continue
            for n in rec.get("news", []) or []:
                title = (n.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                items.append(n)

        return items

    def _build_watchlist_meta(self) -> list[dict]:
        try:
            tickers = self.watchlist_provider() or []
        except Exception:
            tickers = []
        from core.analysis_store import TICKER_NAMES
        return [{"ticker": t, "name": TICKER_NAMES.get(t, t)} for t in tickers]

    def _build_holdings_meta(self) -> list[dict]:
        meta = []
        try:
            total = getattr(self.portfolio, "total_asset", 0) or 1
            for tk, pos in self.portfolio.positions.items():
                meta.append({
                    "ticker": tk,
                    "name": getattr(pos, "name", tk),
                    "strategy_type": getattr(pos, "strategy_type", "swing"),
                    "pnl_pct": round(getattr(pos, "pnl_pct", 0), 4),
                    "weight": round((getattr(pos, "market_value", 0) or 0) / total, 4),
                })
        except Exception as e:
            logger.debug(f"holdings_meta build failed: {e}")
        return meta

    def _build_market_context(self) -> dict:
        """코스피/환율 등 기본 시장 컨텍스트. API 실패 시 빈 dict."""
        ctx: dict = {"as_of": datetime.now().isoformat()}
        try:
            kospi = self.market_client.get_kospi_index() if self.market_client else {}
            if kospi:
                ctx["kospi"] = {
                    "price": kospi.get("price"),
                    "change_pct": kospi.get("change_pct"),
                }
        except Exception as e:
            logger.debug(f"KOSPI context skipped: {e}")
        return ctx

    def _load_previous_implications(self) -> str:
        """직전 분석의 trading_implications — 변화 추적용."""
        cached = self._load_latest()
        if not cached:
            return ""
        return cached.get("trading_implications", "") or ""

    # ============================================================
    # 저장/로드
    # ============================================================

    def _load_latest(self) -> dict | None:
        if not LATEST_CACHE_PATH.exists():
            return None
        try:
            return json.loads(LATEST_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"news_deep_latest.json corrupt: {e}")
            return None

    def _save_latest(self, result: dict) -> None:
        try:
            LATEST_CACHE_PATH.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"news_deep_latest save failed: {e}")

    def _save_report_files(self, result: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        base = REVIEWS_DIR / f"news_deep_{ts}"
        try:
            base.with_suffix(".json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"news_deep JSON save failed: {e}")
        try:
            base.with_suffix(".md").write_text(
                self._format_md(result), encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"news_deep MD save failed: {e}")

    # ============================================================
    # 포맷팅
    # ============================================================

    def _format_md(self, r: dict) -> str:
        lines = [
            f"# 뉴스 심층 분석 — {r.get('timestamp', '')[:16]}",
            "",
            f"- 누적 뉴스: {r.get('news_count', 0)}건 / 윈도우: {r.get('lookback_hours', 4)}시간",
            f"- 헤드라인: {r.get('headline', '')}",
            "",
            "## 거시 환경",
            r.get("macro_summary", ""),
            "",
        ]

        sectors = r.get("sector_trends", []) or []
        if sectors:
            lines.append("## 섹터 흐름")
            for s in sectors:
                tickers = ", ".join(s.get("tickers", []) or [])
                lines.append(f"### {s.get('sector', '')} — {s.get('trend', '')}")
                lines.append(f"- 동인: {s.get('drivers', '')}")
                if tickers:
                    lines.append(f"- 관련 종목: {tickers}")
            lines.append("")

        impacts = r.get("ticker_impacts", []) or []
        if impacts:
            lines.append("## 종목별 영향")
            for i in impacts:
                emoji = {"positive": "📈", "negative": "📉", "neutral": "⚪"}.get(i.get("sentiment", ""), "?")
                lines.append(
                    f"- {emoji} {i.get('name', '')} ({i.get('ticker', '')}) "
                    f"[{i.get('sentiment', '')} / {i.get('impact_score', '?')}/10]"
                )
                lines.append(f"  └ {i.get('reason', '')}")
            lines.append("")

        risks = r.get("risk_events", []) or []
        if risks:
            lines.append("## 리스크 이벤트")
            for risk in risks:
                lines.append(
                    f"- [{risk.get('severity', '?')}] {risk.get('event', '')}"
                )
                affected = ", ".join(risk.get("affected_sectors", []) or [])
                if affected:
                    lines.append(f"  └ 영향 섹터: {affected}")
                hint = risk.get("action_hint", "")
                if hint:
                    lines.append(f"  └ 권고: {hint}")
            lines.append("")

        impl = r.get("trading_implications", "")
        if impl:
            lines.extend(["## 매매 시스템 시그널 (trading_implications)", impl])

        return "\n".join(lines)

    def _send_telegram(self, r: dict) -> None:
        if not self.telegram or not getattr(self.telegram, "enabled", False):
            return

        headline = r.get("headline", "뉴스 심층 분석")
        lines = [
            f"📰 <b>뉴스 심층 분석</b>",
            f"<i>{headline}</i>",
            f"누적 {r.get('news_count', 0)}건 / 최근 {r.get('lookback_hours', 4)}h",
            "",
            "<b>거시 요약</b>",
            (r.get("macro_summary", "") or "")[:400],
        ]

        sectors = r.get("sector_trends", []) or []
        if sectors:
            lines.append("")
            lines.append("<b>섹터 흐름</b>")
            for s in sectors[:5]:
                trend = s.get("trend", "")
                lines.append(f"• {s.get('sector', '')} — {trend}: {(s.get('drivers', '') or '')[:80]}")

        impacts = r.get("ticker_impacts", []) or []
        positive = [i for i in impacts if (i.get("sentiment") or "").lower() == "positive"]
        negative = [i for i in impacts if (i.get("sentiment") or "").lower() == "negative"]
        if positive:
            lines.append("")
            lines.append("<b>📈 긍정 종목</b>")
            for i in sorted(positive, key=lambda x: -(x.get("impact_score") or 0))[:4]:
                lines.append(
                    f"• {i.get('name', '')}({i.get('ticker', '')}) "
                    f"[{i.get('impact_score', '?')}/10] {(i.get('reason', '') or '')[:100]}"
                )
        if negative:
            lines.append("")
            lines.append("<b>📉 부정 종목</b>")
            for i in sorted(negative, key=lambda x: -(x.get("impact_score") or 0))[:4]:
                lines.append(
                    f"• {i.get('name', '')}({i.get('ticker', '')}) "
                    f"[{i.get('impact_score', '?')}/10] {(i.get('reason', '') or '')[:100]}"
                )

        risks = r.get("risk_events", []) or []
        high_risks = [r for r in risks if (r.get("severity") or "").lower() == "high"]
        if high_risks:
            lines.append("")
            lines.append("<b>🔴 High 리스크</b>")
            for risk in high_risks[:3]:
                lines.append(f"• {(risk.get('event', '') or '')[:150]}")

        impl = r.get("trading_implications", "")
        if impl:
            lines.append("")
            lines.append("<b>매매 시스템 시그널</b>")
            lines.append(impl[:500])

        try:
            self.telegram.send_alert_sync("news_deep", "\n".join(lines))
        except Exception as e:
            logger.warning(f"News deep telegram send failed: {e}")

    def _notify_failure(self, reason: str) -> None:
        if not self.telegram or not getattr(self.telegram, "enabled", False):
            return
        try:
            self.telegram.send_alert_sync(
                "error",
                f"⚠️ <b>News Deep Analysis 실패</b>\n사유: {reason[:300]}",
            )
        except Exception:
            pass
