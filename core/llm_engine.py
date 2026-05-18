"""LLM 전략 엔진 — Claude API 연동, 매매 판단, 전략 해석, config 자동 조정."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"
DECISIONS_LOG_DIR = Path(__file__).parent.parent / "logs" / "decisions"


class LLMEngine:
    """시스템의 두뇌 — 모든 매매 판단의 최종 결정자."""

    def __init__(self, config: dict):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        llm_config = config.get("llm", {})
        self.model = llm_config.get("model", "claude-sonnet-4-6")
        self.model_deep = llm_config.get("model_deep", "claude-opus-4-6")  # 2단계 깊이 분석용
        self.max_tokens = llm_config.get("max_tokens", 2048)
        self.temperature = llm_config.get("temperature", 0.3)
        # 카테고리별 일일 한도 — 한 기능 폭주가 다른 기능 차단하지 못하도록 분리
        cost_limits = llm_config.get("cost_limits_daily_usd", {})
        # backward compat: 기존 cost_limit_daily_usd 키는 trading 카테고리로 매핑
        legacy_limit = llm_config.get("cost_limit_daily_usd")
        self.cost_limits: dict[str, float] = {
            "trading": cost_limits.get("trading", legacy_limit if legacy_limit is not None else 10.0),
            "daily_review": cost_limits.get("daily_review", 1.0),
            "weekly_review": cost_limits.get("weekly_review", 15.0),
            "rca": cost_limits.get("rca", 5.0),
            "news": cost_limits.get("news", 5.0),
        }
        self._daily_costs: dict[str, float] = {k: 0.0 for k in self.cost_limits}
        self._cost_date = ""
        self.force_include_tickers: list[str] = []
        DECISIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def set_force_include_tickers(self, tickers: list[str]) -> None:
        """include_tickers 강제 후보화 — 보유 안 한 종목에 한해 stage 1 candidates에 자동 주입."""
        self.force_include_tickers = list(tickers or [])

    def _reset_daily_costs_if_new_day(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cost_date != today:
            self._daily_costs = {k: 0.0 for k in self.cost_limits}
            self._cost_date = today

    def _check_cost_limit(self, category: str = "trading") -> bool:
        self._reset_daily_costs_if_new_day()
        limit = self.cost_limits.get(category, self.cost_limits["trading"])
        spent = self._daily_costs.get(category, 0.0)
        if spent >= limit:
            logger.warning(f"LLM daily cost limit reached [{category}]: ${spent:.2f} >= ${limit:.2f}")
            return False
        return True

    # backward compat property
    @property
    def cost_limit_daily(self) -> float:
        return self.cost_limits["trading"]

    @property
    def _daily_cost(self) -> float:
        """전체 누적 비용 (모든 카테고리 합)."""
        return sum(self._daily_costs.values())

    # 모델별 가격 (USD per 1M tokens). 새 모델 추가 시 여기에 등록.
    _MODEL_PRICING = {
        "claude-opus-4-7": (15.0, 75.0),
        "claude-opus-4-6": (15.0, 75.0),
        "claude-sonnet-4-6": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
    }

    @classmethod
    def _resolve_pricing(cls, model: str) -> tuple[float, float]:
        """모델 이름 → (input_price, output_price) per 1M tokens. 매핑 없으면 Sonnet 기준 fallback."""
        # 정확 매칭 우선, 그 다음 prefix 매칭 (예: claude-opus-4-7-2026... 같은 변종 대응)
        if model in cls._MODEL_PRICING:
            return cls._MODEL_PRICING[model]
        for known, prices in cls._MODEL_PRICING.items():
            if model.startswith(known):
                return prices
        logger.warning(f"Unknown model pricing for {model}; falling back to Sonnet rates")
        return (3.0, 15.0)

    def _track_cost(self, usage: dict, model: str, category: str = "trading") -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        in_price, out_price = self._resolve_pricing(model)
        cost = (input_tokens * in_price / 1_000_000) + (output_tokens * out_price / 1_000_000)
        self._reset_daily_costs_if_new_day()
        self._daily_costs[category] = self._daily_costs.get(category, 0.0) + cost

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        category: str = "trading",
        max_tokens: int | None = None,
    ) -> str:
        if not self._check_cost_limit(category):
            raise RuntimeError(f"LLM daily cost limit reached [{category}]")

        use_model = model or self.model
        # Opus 4.7부터 temperature 파라미터 deprecated (400 에러). SDK default 사용.
        response = self.client.messages.create(
            model=use_model,
            max_tokens=max_tokens or self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        self._track_cost({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }, model=use_model, category=category)

        return response.content[0].text

    def load_strategy(self) -> str:
        active_path = STRATEGIES_DIR / "active.md"
        if not active_path.exists():
            logger.warning("No active strategy found, using default")
            return "기본 추세추종 전략. RSI 과매도 매수, 과매수 매도."
        return active_path.read_text(encoding="utf-8")

    def analyze_market(
        self,
        portfolio: dict,
        market_data: dict,
        ml_predictions: dict | None = None,
        news_summary: str = "",
        macro_data: dict | None = None,
        screening_context: str = "",
        prediction_feedback: str = "",
        backtest_feedback: str = "",
    ) -> dict:
        """시장 분석 및 매매 판단 — 2단계 분석.

        1단계: 빠른 분류 (8종목+ 관심종목 → SELL/HOLD/CANDIDATE)
        2단계: CANDIDATE 종목만 깊이 분석 → BUY 또는 HOLD 결정
        """
        strategy = self.load_strategy()

        # 1단계: 빠른 필터링
        triage = self._stage1_triage(
            strategy, portfolio, market_data, ml_predictions or {},
            news_summary, screening_context,
        )

        # 1단계 결과에서 즉시 처리(SELL) + 후보(CANDIDATE) 분리
        immediate_actions = triage.get("immediate_actions", [])
        candidates = triage.get("candidates", [])

        # include_tickers 강제 후보화 — 보유 안 한 종목은 무조건 stage 2 분석 보장 (LLM이 빠뜨려도 강제 주입)
        if self.force_include_tickers:
            held = set(portfolio.get("positions", {}).keys())
            for t in self.force_include_tickers:
                if t not in held and t not in candidates:
                    candidates.append(t)
                    logger.info(f"Stage 1 candidates 강제 주입: {t} (include_tickers)")

        # 2단계: 후보 종목만 깊이 분석
        deep_actions = []
        deep = {}
        if candidates:
            deep = self._stage2_deep_analysis(
                strategy, portfolio, market_data, ml_predictions or {},
                news_summary, macro_data or {}, candidates,
                prediction_feedback, backtest_feedback,
            )
            # 2단계에서는 BUY/SELL만 필터링 (HOLD는 제외)
            deep_actions = [a for a in deep.get("actions", []) if a.get("type", "").upper() in ("BUY", "SELL")]

        # 결과 통합
        all_actions = immediate_actions + deep_actions

        # HOLD 종목 자동 추가 (보유 중인데 액션이 없는 종목)
        action_tickers = {a.get("ticker") for a in all_actions}
        hold_reasons = triage.get("hold_reasons", {}) if isinstance(triage.get("hold_reasons"), dict) else {}
        for ticker in portfolio.get("positions", {}):
            if ticker not in action_tickers:
                pos = portfolio["positions"][ticker]
                all_actions.append({
                    "type": "HOLD",
                    "ticker": ticker,
                    "name": pos.get("name", ticker),
                    "ratio": 0.0,
                    "reason": hold_reasons.get(ticker, "보유 유지"),
                })

        signal = {
            "actions": all_actions,
            "reasoning": triage.get("reasoning", "") + (" | " + deep.get("reasoning", "") if deep else ""),
            "risk_assessment": triage.get("risk_assessment", "MEDIUM"),
            "market_outlook": triage.get("market_outlook", ""),
            "config_adjustments": [],
        }

        # 판단 근거 로깅
        self._log_decision(signal, json.dumps(triage, ensure_ascii=False) + "\n---\n" + json.dumps(deep_actions, ensure_ascii=False))

        return signal

    def _stage1_triage(
        self, strategy: str, portfolio: dict, market_data: dict,
        ml_predictions: dict, news_summary: str, screening_context: str,
    ) -> dict:
        """1단계: 빠른 분류 — SELL 즉시 결정 + CANDIDATE 선별."""
        system_prompt = """당신은 한국 주식 매매 시스템의 1단계 분류기입니다.
주어진 종목들을 빠르게 분류만 합니다. 깊이 분석은 2단계에서 합니다.

매도 가능 조건 (사실 기반만):
- 손절선 도달 (value -10%, swing -5%, daytrading -3%)
- daytrading +5% 도달 — 스캘프 즉시 매도
- 트레일링 스탑 (수익 구간에서 고점 대비 -5~10%, 섹터별)
- ⚠️ swing +10% / value +20% 도달은 즉시 매도 사유 아님 — 트레일링이 자동 관리하므로 HOLD 유지.
  예외: 익절 도달 + 당일 음봉 + 거래량 평소 2배 이상 (분배 시그널) → SELL 허용

⚠️ 포지션 교체(Rotation) 규칙:
- 보유종목이 max_positions에 도달해도, 더 좋은 매수 후보가 있으면 교체 가능
- 보유종목 중 rotation_score가 가장 낮고 0 미만인 종목 = 교체 대상
- 교체 대상은 immediate_actions에 SELL로 넣고, 대체할 종목을 candidates에 넣을 것
- value 종목은 교체 금지

⚠️ candidates를 반드시 뽑아라:
- 워치리스트 중 현재 미보유이면서 매수 조건에 부합하는 종목을 최대 5개 선별
- 보유 포화(max_positions) 상태라도 candidates는 반드시 뽑아야 함 (포지션 교체용)
- candidates가 비어있으면 2단계 분석이 실행되지 않아 매수 기회를 영원히 놓침

JSON 형식으로 응답:
{
  "reasoning": "전체 시장 한 줄 요약",
  "market_outlook": "시장 전망",
  "risk_assessment": "LOW | MEDIUM | HIGH",
  "immediate_actions": [
    {"type": "SELL", "ticker": "종목코드", "name": "종목명", "ratio": 1.0, "reason": "매도 사유"}
  ],
  "candidates": ["종목코드1", "종목코드2"],
  "hold_reasons": {
    "종목코드": "보유 유지 근거 한 줄 (수익률, 추세, 수급 등 사실 기반)"
  }
}

immediate_actions: 즉시 매도해야 할 보유종목 (사실 기반만 + 교체 대상)
candidates: 매수 검토 가치가 있는 워치리스트 종목 (최대 5개, 반드시 뽑을 것)
hold_reasons: immediate_actions에 없는 모든 보유종목에 대해 한 줄 이유 (왜 HOLD인지)"""

        user_message = self._build_triage_prompt(
            strategy, portfolio, market_data, ml_predictions, news_summary, screening_context,
        )
        raw = self._call_llm(system_prompt, user_message)
        return self._parse_json_response(raw) or {}

    def _stage2_deep_analysis(
        self, strategy: str, portfolio: dict, market_data: dict,
        ml_predictions: dict, news_summary: str, macro_data: dict,
        candidates: list[str], prediction_feedback: str, backtest_feedback: str,
    ) -> dict:
        """2단계: 후보 종목 깊이 분석 → BUY 결정."""
        # 후보 종목만 필터링
        candidate_market = {t: market_data.get(t, {}) for t in candidates if t in market_data}
        candidate_ml = {t: ml_predictions.get(t, {}) for t in candidates if t in ml_predictions}

        system_prompt = self._build_system_prompt()
        user_message = self._build_analysis_prompt(
            strategy=strategy,
            portfolio=portfolio,
            market_data=candidate_market,
            ml_predictions=candidate_ml,
            news_summary=news_summary,
            macro_data=macro_data,
            screening_context=f"## 2단계 분석 대상\n매수 후보로 1단계에서 선별된 종목들입니다: {candidates}",
            prediction_feedback=prediction_feedback,
            backtest_feedback=backtest_feedback,
        )
        raw = self._call_llm(system_prompt, user_message, model=self.model_deep)
        logger.info(f"Stage 2 deep analysis used model: {self.model_deep}")
        return self._parse_signal(raw)

    def _build_triage_prompt(
        self, strategy: str, portfolio: dict, market_data: dict,
        ml_predictions: dict, news_summary: str, screening_context: str,
    ) -> str:
        """1단계 분류용 간단한 프롬프트."""
        parts = [
            "## 전략 (요약)",
            strategy[:2000],
            "",
            "## 포트폴리오",
            json.dumps(portfolio, ensure_ascii=False, indent=2),
            "",
        ]
        if screening_context:
            parts.extend([screening_context, ""])
        parts.extend([
            "## 시장 데이터",
            json.dumps(market_data, ensure_ascii=False, indent=2),
            "",
        ])
        if ml_predictions:
            # ML은 요약만
            ml_summary = {t: {"prediction": v.get("prediction", ""), "confidence": v.get("confidence", 0)}
                          for t, v in ml_predictions.items()}
            parts.extend([
                "## ML 예측 (요약)",
                json.dumps(ml_summary, ensure_ascii=False, indent=2, default=str),
                "",
            ])
        if news_summary:
            parts.extend(["## 뉴스 요약", news_summary[:1000], ""])
        parts.append("위 정보를 빠르게 분류하세요. immediate_actions에는 즉시 매도할 보유종목, candidates에는 매수 검토할 종목 코드만 나열하세요.")
        return "\n".join(parts)

    def interpret_natural_language(self, user_input: str, current_params: dict) -> dict:
        """자연어 전략 변경 → config 조정 목록 반환."""
        system_prompt = """당신은 AI 트레이딩 시스템의 설정 관리자입니다.
사용자의 자연어 지시를 해석하여 매매 파라미터를 조정합니다.

현재 조정 가능한 파라미터:
- holding_period_days.min / holding_period_days.max: 보유 기간
- trailing_stop_pct: 트레일링 스톱 (null이면 미사용)
- position_size_pct: 포지션 크기 (0.03~0.15)
- max_cash_ratio: 최소 현금 비중 (0.10~0.90)
- indicators.rsi_oversold / indicators.rsi_overbought: RSI 기준
- indicators.volume_surge_multiplier: 거래량 급증 배수
- indicators.ma_short / indicators.ma_long: 이동평균 기간
- portfolio_targets.monthly_target_pct: 월 목표 수익률
- portfolio_targets.on_target_reached: 목표 도달 시 행동
- market_conditions.vix_high_threshold: VIX 경계값

반드시 JSON 형식으로 응답하세요:
{
  "interpretation": "사용자 지시 해석 (한국어)",
  "adjustments": [
    {"param": "파라미터명", "value": 새값, "reason": "조정 이유"}
  ],
  "strategy_change_needed": false,
  "strategy_suggestion": ""
}"""

        user_message = f"""현재 파라미터:
{json.dumps(current_params, ensure_ascii=False, indent=2)}

사용자 지시: {user_input}

위 지시를 해석하여 파라미터 조정 목록을 JSON으로 반환하세요."""

        raw = self._call_llm(system_prompt, user_message)
        return self._parse_json_response(raw)

    def generate_review(
        self,
        trades: list[dict],
        portfolio_change: dict,
        market_summary: dict,
    ) -> dict:
        """일일 자기 복기."""
        system_prompt = """당신은 AI 트레이딩 시스템의 자기 복기 엔진입니다.
오늘의 매매를 분석하고 개선점을 제안합니다.

반드시 JSON 형식으로 응답하세요:
{
  "summary": "오늘의 전체 평가 (2-3문장)",
  "trade_reviews": [
    {
      "ticker": "종목코드",
      "action": "BUY/SELL",
      "evaluation": "적절/부적절/판단보류",
      "timing_score": 1-10,
      "size_score": 1-10,
      "selection_score": 1-10,
      "comment": "개별 평가"
    }
  ],
  "improvements": ["개선점1", "개선점2"],
  "strategy_modification": {
    "needed": false,
    "suggestion": "",
    "reason": ""
  },
  "overall_score": 1-10,
  "market_insight": "시장 인사이트 한 줄"
}"""

        user_message = f"""오늘의 매매 이력:
{json.dumps(trades, ensure_ascii=False, indent=2)}

포트폴리오 변동:
{json.dumps(portfolio_change, ensure_ascii=False, indent=2)}

시장 요약:
{json.dumps(market_summary, ensure_ascii=False, indent=2)}

위 데이터를 기반으로 복기해주세요."""

        raw = self._call_llm(system_prompt, user_message)
        return self._parse_json_response(raw)

    def generate_deep_review(
        self,
        trades: list[dict],
        decisions: list[dict],
        safety_stats: dict,
        missed_opportunities: list[dict],
        portfolio_change: dict,
        portfolio_summary: dict,
        market_summary: dict,
        strategy_excerpt: str = "",
    ) -> dict:
        """장 마감 후 Opus 4.7로 심층 복기 — 패턴/메타인지/룰 충돌 분석.

        category='daily_review'로 비용 추적. Opus 4.7 사용 (model_deep 무시하고 명시 지정).
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 일일 심층 복기 엔진입니다.
오늘의 매매, 결정 로그, 룰 적용 결과, 놓친 기회를 종합 분석합니다.

분석 포커스:
1. 룰 적용 일관성 — 동일/유사 시그널에 대해 BUY vs HOLD 판단이 일관적이었는가
2. 놓친 기회 (missed_opportunities) — 스크리닝 상위인데 매수 안 한 종목의 보류 사유가 합리적이었는지
3. 룰 충돌 (safety_filtered/circuit_blocked) — 자동 차단이 정확했는지, 아니면 과보수였는지
4. 포트폴리오 균형 — 섹터 집중, 현금 비율, 일일 PnL 추이
5. 액션 가능 개선점 — 내일 즉시 적용 가능한 룰 미세 조정

반드시 아래 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만:
{
  "headline": "텔레그램 헤더용 한 줄 요약 (80자 이내)",
  "overall_score": 1-10,
  "key_findings": ["발견사항1", "발견사항2", "발견사항3"],
  "rule_consistency": {
    "score": 1-10,
    "notes": "BUY/HOLD 판단 일관성 분석 (2-3문장)"
  },
  "missed_opportunities": [
    {"ticker": "...", "name": "...", "reason_not_bought": "...", "evaluation": "합리적/아쉬움/명백한 실수"}
  ],
  "rule_conflicts": [
    {"rule": "safety_guard 또는 circuit_breaker 또는 룰명", "blocked": "차단된 액션 설명", "verdict": "정확/과보수/검토필요"}
  ],
  "portfolio_balance": "섹터/현금/리스크 평가 (2-3문장)",
  "improvements": ["내일 즉시 적용 가능한 개선점1", "개선점2"],
  "strategy_modification_hint": {
    "needed": false,
    "rule_id": "active.md의 어느 섹션",
    "current": "현재 룰 발췌",
    "suggestion": "수정 제안 (구체적으로)"
  }
}"""

        # 결정 로그는 토큰 절감을 위해 핵심만 추출
        compact_decisions = []
        for d in decisions:
            sig = d.get("signal", {})
            compact_decisions.append({
                "ts": d.get("timestamp", "")[:16],
                "risk": sig.get("risk_assessment", ""),
                "outlook": (sig.get("market_outlook", "") or "")[:200],
                "reasoning": (sig.get("reasoning", "") or "")[:500],
                "actions": [
                    {
                        "type": a.get("type"), "ticker": a.get("ticker"),
                        "ratio": a.get("ratio"),
                        "reason": (a.get("reason", "") or "")[:200],
                    }
                    for a in sig.get("actions", [])
                ],
                "safety_filtered": sig.get("safety_filtered", []) or [],
            })

        user_message = f"""## 오늘 시장 요약
{json.dumps(market_summary, ensure_ascii=False)}

## 포트폴리오 변동
{json.dumps(portfolio_change, ensure_ascii=False)}

## 현재 포트폴리오 상태
{json.dumps(portfolio_summary, ensure_ascii=False)}

## 오늘 매매 ({len(trades)}건)
{json.dumps(trades, ensure_ascii=False, indent=2)[:8000]}

## 결정 로그 요약 ({len(compact_decisions)}개 사이클)
{json.dumps(compact_decisions, ensure_ascii=False, indent=2)[:20000]}

## 룰 차단 통계
{json.dumps(safety_stats, ensure_ascii=False)}

## 놓친 기회 후보
{json.dumps(missed_opportunities, ensure_ascii=False)}

## 활성 전략 발췌 (참고)
{strategy_excerpt[:3000]}

위 데이터를 기반으로 일일 심층 복기를 JSON으로만 응답해주세요."""

        raw = self._call_llm(
            system_prompt, user_message,
            model="claude-opus-4-7",  # Opus 명시 지정
            category="daily_review",
        )
        return self._parse_json_response(raw)

    def generate_incident_rca(
        self,
        event_type: str,
        ticker: str | None,
        event_detail: str,
        context: dict,
    ) -> dict:
        """사고 RCA — Opus 4.7로 근본 원인 분석.

        category='rca'로 비용 추적. event_type 예: daily_loss_2pct, circuit_breaker_halted,
        sell_backoff_blocked, hard_stop_loss.
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 사고 분석 엔진(Incident RCA)입니다.
이벤트 발생 시점의 시스템 상태를 토대로 근본 원인을 찾아냅니다.

분석 절차:
1. 이벤트 타임라인 재구성 — 최근 결정/매매에서 사고 직전 시그널이 있었는지
2. 근본 원인 — 룰 미비, LLM 오판, 시장 변동, 운영 사고 중 어느 카테고리인가
3. 기여 요인 — 누적된 결정/매매 패턴 중 사고를 키운 요소
4. 재발 방지 — 즉시 가능한 운영 조치 + 룰/전략 수정 제안

반드시 아래 JSON 형식으로만 응답하세요:
{
  "headline": "한 줄 요약 (80자 이내, 텔레그램용)",
  "severity": "low | medium | high",
  "root_cause": "근본 원인 (2-3문장, 한국어)",
  "contributing_factors": ["기여요인1", "기여요인2"],
  "timeline": [
    {"when": "HH:MM", "what": "어떤 결정/매매/이벤트", "verdict": "정상/문제/모름"}
  ],
  "preventive_actions": [
    {"action": "즉시 또는 단기에 적용할 조치", "scope": "ops | rule | strategy"}
  ],
  "rule_change_suggestion": {
    "needed": true,
    "rule_id": "active.md 어느 섹션 또는 trading-params.yaml의 어느 키",
    "current": "현재 동작 발췌",
    "suggestion": "수정 제안 (구체적으로)"
  }
}"""

        user_message = f"""## 사고 정보
- 유형: {event_type}
- 종목: {ticker or 'N/A'}
- 상세: {event_detail}
- 발생 시각: {datetime.now().isoformat()}

## 시스템 컨텍스트
{json.dumps(context, ensure_ascii=False, indent=2, default=str)[:40000]}

위 정보를 기반으로 근본 원인 분석을 JSON으로만 응답해주세요."""

        raw = self._call_llm(
            system_prompt, user_message,
            model="claude-opus-4-7",
            category="rca",
        )
        return self._parse_json_response(raw)

    def generate_weekly_refinement(
        self,
        trades_week: list[dict],
        decisions_week: list[dict],
        backtest_summary: dict,
        safety_stats: dict,
        portfolio_summary: dict,
        weekly_pnl_summary: dict,
        current_active_md: str,
    ) -> dict:
        """주간 전략 다듬기 — Opus 4.7로 active.md 전문 + 변경 요약 출력.

        category='weekly_review'로 비용 추적 ($15/일 한도).
        자동 적용 절대 없음 — 호출자가 텔레그램 승인 후 적용.
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 주간 전략 다듬기(Strategy Refinement) 엔진입니다.
지난 7거래일의 매매·결정·룰 차단·백테스트 결과를 종합 분석하여 현재 활성 전략(active.md)을
개선합니다. 핵심 철학과 안전장치 우선순위는 유지하되, 데이터로 검증된 미세 조정만 반영합니다.

분석 원칙:
1. **데이터 기반** — 가설이 아니라 실제 매매·결정 패턴에서 검증된 부분만 수정
2. **점진 개선** — 한 번에 전략을 뒤집지 말 것. 좁고 명확한 룰 수정 위주
3. **충돌 해소** — safety_filtered 통계에서 자주 차단된 룰은 LLM 판단과 충돌. 어느 쪽이 옳은지 판단
4. **놓친 기회 vs 손실** — 어느 쪽이 더 큰지 가중 분석. 한 쪽 사례만 보고 룰을 풀거나 조이지 말 것
5. **핵심 안전장치 유지** — 손절선/트레일링/시장가 금지/F~J 가격대 룰 같은 안전장치는 절대 약화 금지
6. **strategy_type별 분리** — value/swing/daytrading 룰을 섞지 말 것
7. **새 active.md 전문 출력** — 부분 패치가 아니라 새 버전 전체 (md 형식 그대로)

⚠️ 금지:
- 손절 폭 -5%/-10% 완화 금지
- 시장가 매수 허용 금지 (`urgency: limit` 강제 유지)
- 익절 단독 매도 허용 금지 (트레일링 자동 관리 원칙 유지)
- 신규 strategy_type 도입 금지
- 미보유 종목 SELL 룰(E3) 약화 금지

반드시 아래 JSON 형식으로만 응답하세요:
{
  "headline": "한 줄 요약 (80자 이내, 텔레그램용)",
  "key_observations": ["이번 주 관찰1", "관찰2", "관찰3"],
  "what_worked": ["성공 패턴1", "패턴2"],
  "what_failed": ["실패 패턴1", "패턴2"],
  "proposed_changes": [
    {
      "section": "active.md의 어느 섹션 (예: 스윙 매수 조건, F~J 가격대 룰)",
      "current": "현재 룰 발췌 (한 줄)",
      "new": "수정안 (한 줄)",
      "rationale": "데이터 근거 (어느 매매/결정/통계에서 도출했는지)",
      "expected_impact": "기대 효과 (한 줄)"
    }
  ],
  "new_active_md": "새 active.md 전문 (markdown 형식 그대로, 최소 2000자, '# 트레이딩 전략' 헤더 포함)",
  "risk_notes": "변경 시 주의사항 / 모니터링 포인트"
}"""

        # 결정 로그는 토큰 절감용 compact
        compact_decisions = []
        for d in decisions_week[-150:]:  # 7일 분량 cap
            sig = d.get("signal", {})
            compact_decisions.append({
                "ts": d.get("timestamp", "")[:16],
                "risk": sig.get("risk_assessment", ""),
                "reasoning": (sig.get("reasoning", "") or "")[:300],
                "actions": [
                    {
                        "type": a.get("type"), "ticker": a.get("ticker"),
                        "strategy_type": a.get("strategy_type"),
                        "ratio": a.get("ratio"),
                        "reason": (a.get("reason", "") or "")[:150],
                    }
                    for a in sig.get("actions", [])
                ],
                "safety_filtered": sig.get("safety_filtered", []) or [],
            })

        user_message = f"""## 지난 7거래일 매매 ({len(trades_week)}건)
{json.dumps(trades_week, ensure_ascii=False, indent=2)[:20000]}

## 결정 로그 요약 ({len(compact_decisions)}건)
{json.dumps(compact_decisions, ensure_ascii=False, indent=2)[:40000]}

## 룰 차단 통계 (safety_guard + circuit_breaker)
{json.dumps(safety_stats, ensure_ascii=False, indent=2)}

## 주간 PnL 요약
{json.dumps(weekly_pnl_summary, ensure_ascii=False, indent=2)}

## 현재 포트폴리오 상태
{json.dumps(portfolio_summary, ensure_ascii=False, indent=2)}

## 백테스트 결과 요약
{json.dumps(backtest_summary, ensure_ascii=False, indent=2)[:8000]}

## 현재 active.md 전문
{current_active_md}

위 데이터를 기반으로 주간 전략 다듬기를 JSON으로만 응답해주세요.
new_active_md는 반드시 완전한 markdown 전문을 포함해야 합니다 (최소 2000자)."""

        raw = self._call_llm(
            system_prompt, user_message,
            model="claude-opus-4-7",
            category="weekly_review",
            max_tokens=16384,
        )
        return self._parse_json_response(raw)

    def extract_param_adjustment(
        self,
        suggestion_text: str,
        rule_id: str,
        current_params_flat: dict,
        adjustable_limits: dict,
    ) -> dict:
        """자연어 제안을 trading-params.yaml의 {param, value, reason}로 추출.

        category='trading'으로 비용 추적 (작은 호출, Sonnet 충분).
        반환: {"param": "...", "value": ..., "reason": "..."} 또는 {"error": "..."}.
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 룰 수정 보조 엔진입니다.
자연어 제안을 trading-params.yaml의 단일 키-값 수정 형식으로 변환합니다.

규칙:
1. **adjustable_limits 화이트리스트 내 파라미터만 출력** — 목록 밖이면 error
2. **단일 키 수정만** — 여러 키 동시 수정 제안은 가장 핵심 1개만 선택
3. **현재값 대비 합리적 범위** — 극단적 변경(>50%) 회피
4. **active.md 수정이 필요한 제안이면 error** — "active.md 텍스트 수정 필요"로 분류

반드시 JSON으로만 응답:
- 성공: {"param": "param.key", "value": 숫자또는불린, "reason": "변경 이유 한 줄"}
- 실패: {"error": "에러 사유 (한 줄)"}"""

        user_message = f"""## 제안 원문
- rule_id: {rule_id}
- suggestion: {suggestion_text}

## 현재 trading-params (플랫)
{json.dumps(current_params_flat, ensure_ascii=False, indent=2)[:3000]}

## adjustable_limits (화이트리스트 + 범위)
{json.dumps(adjustable_limits, ensure_ascii=False, indent=2)[:2000]}

JSON으로만 응답."""

        raw = self._call_llm(system_prompt, user_message, category="trading", max_tokens=1024)
        return self._parse_json_response(raw)

    def apply_suggestion_to_strategy(
        self,
        suggestion_text: str,
        rule_id: str,
        current_active_md: str,
    ) -> dict:
        """단일 제안을 active.md에 반영한 새 전문 생성. weekly_refinement와 유사하나
        한 가지 제안에만 집중. category='weekly_review'로 비용 추적.

        반환: weekly_refinement와 동일 구조 (headline, new_active_md, proposed_changes, ...).
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 전략 다듬기 엔진입니다.
사용자가 텔레그램에서 daily_review/RCA 제안을 검토용으로 변환 요청했습니다.
**한 가지 제안에만** 집중하여 active.md를 미세 조정한 새 전문을 출력합니다.

원칙:
1. **제안 범위만 수정** — 다른 섹션은 그대로 유지
2. **핵심 안전장치 유지** — 손절선/시장가 금지/E3 미보유 SELL 룰 절대 약화 금지
3. **새 active.md 전문 출력** — 부분 패치가 아니라 새 버전 전체

⚠️ 금지:
- 손절 폭 -5%/-10% 완화 금지
- 시장가 매수 허용 금지 (urgency: limit 강제 유지)
- 익절 단독 매도 허용 금지
- 새 strategy_type 도입 금지
- E3(미보유 SELL) 룰 약화 금지

반드시 JSON 형식:
{
  "headline": "한 줄 요약 (80자 이내, 텔레그램용)",
  "proposed_changes": [
    {"section": "...", "current": "...", "new": "...", "rationale": "..."}
  ],
  "new_active_md": "새 active.md 전문 (markdown 형식, 최소 2000자, '# 트레이딩 전략' 헤더 포함)",
  "risk_notes": "모니터링 포인트"
}"""

        user_message = f"""## 검토용 변환 요청 (단일 제안)
- rule_id: {rule_id}
- suggestion: {suggestion_text}

## 현재 active.md 전문
{current_active_md}

위 제안만 반영한 새 active.md 전문을 JSON으로만 응답하세요.
new_active_md는 반드시 완전한 markdown 전문 (최소 2000자)."""

        raw = self._call_llm(
            system_prompt, user_message,
            model="claude-opus-4-7",
            category="weekly_review",  # 같은 카테고리 — Opus + active.md 전문 생성
            max_tokens=16384,
        )
        return self._parse_json_response(raw)

    def generate_news_deep_analysis(
        self,
        news_items: list[dict],
        watchlist_meta: list[dict],
        holdings_meta: list[dict],
        market_context: dict,
        previous_implications: str = "",
    ) -> dict:
        """뉴스 심층 분석 — Opus 4.7로 누적 뉴스에서 거시/섹터/종목 시그널 추출.

        category='news'로 비용 추적. 결과의 trading_implications는 다음 trading 사이클의
        news_summary 자리에 주입되어 LLM의 컨텍스트가 됨.
        """
        system_prompt = """당신은 한국 주식 자동매매 시스템의 뉴스 심층 분석 엔진입니다.
지난 4시간 누적된 뉴스 헤드라인을 종합하여 거시 환경, 섹터 흐름, 종목별 영향도,
리스크 이벤트를 정리합니다. 결과는 자동매매 시스템의 다음 매매 사이클이 직접 참조합니다.

분석 원칙:
1. **헤드라인만으로 단정 금지** — 본문이 없으므로 추정에 의존. 확실하지 않은 시그널은 "추정" 명시
2. **워치리스트/보유 종목 우선** — 무관한 종목 분석에 토큰 낭비 금지
3. **거시 → 섹터 → 종목 순서로 좁히기** — 환율/금리/원자재 같은 거시 흐름이 어느 섹터에 영향
4. **리스크 이벤트 강조** — 회계 의혹, 정책 변화, 해외 충격은 별도 risk_events로 분리
5. **trading_implications는 매매 시스템용 실행 가능 시그널** — "어느 섹터 추가 관심", "어느 종목 단기 회피" 등
6. 직전 분석(있을 때)과의 변화에 주목 — 새로운 시그널 vs 지속/약화

⚠️ 절대 금지:
- 헤드라인에 명시되지 않은 종목명/숫자 창작
- "확실히 상승할 것" 같은 단정 — 항상 확률/조건부 표현
- 손절선/익절선 같은 매매 룰 제안 (trading LLM이 별도 판단)

반드시 아래 JSON 형식으로만 응답하세요:
{
  "headline": "텔레그램 헤더 (80자 이내, 가장 중요한 한 줄)",
  "macro_summary": "거시 환경 요약 (환율/금리/지수/대외 충격, 2-3문장)",
  "sector_trends": [
    {
      "sector": "반도체 | 2차전지 | 자동차 | 금융 | ...",
      "trend": "강세 | 약세 | 중립 | 변동성 확대",
      "drivers": "주요 동인 (1-2문장)",
      "tickers": ["관련 워치리스트/보유 종목 코드"]
    }
  ],
  "ticker_impacts": [
    {
      "ticker": "종목코드",
      "name": "종목명",
      "sentiment": "positive | negative | neutral",
      "impact_score": 1-10,
      "reason": "근거 헤드라인 인용 + 한 줄 해석"
    }
  ],
  "risk_events": [
    {
      "event": "이벤트 요약 (1문장)",
      "severity": "low | medium | high",
      "affected_sectors": ["영향받는 섹터"],
      "action_hint": "회피/관망/모니터링 등 한 줄"
    }
  ],
  "trading_implications": "다음 매매 사이클 LLM이 참고할 핵심 시그널 (3-5문장, 한국어). 거시 톤, 강조 섹터, 회피 종목, 리스크 신호를 압축."
}"""

        # 뉴스 압축 (토큰 절감)
        compact_news = []
        for n in news_items[-200:]:  # 최대 200건
            compact_news.append({
                "title": (n.get("title", "") or "")[:200],
                "ticker": n.get("ticker", ""),
                "source": n.get("source", ""),
                "ts": (n.get("datetime", "") or n.get("collected_at", ""))[:16],
            })

        user_message = f"""## 분석 대상 뉴스 ({len(compact_news)}건, 최근 4시간 누적)
{json.dumps(compact_news, ensure_ascii=False, indent=2)[:60000]}

## 워치리스트 ({len(watchlist_meta)}종목)
{json.dumps(watchlist_meta, ensure_ascii=False, indent=2)[:6000]}

## 보유 종목 ({len(holdings_meta)}종목)
{json.dumps(holdings_meta, ensure_ascii=False, indent=2)[:6000]}

## 시장 컨텍스트
{json.dumps(market_context, ensure_ascii=False, indent=2)[:4000]}

## 직전 분석의 trading_implications (있을 때, 변화 비교용)
{previous_implications[:2000] if previous_implications else '(첫 분석)'}

위 데이터를 기반으로 뉴스 심층 분석을 JSON으로만 응답해주세요."""

        raw = self._call_llm(
            system_prompt, user_message,
            model="claude-opus-4-7",
            category="news",
            max_tokens=8192,
        )
        return self._parse_json_response(raw)

    def _build_system_prompt(self) -> str:
        return """당신은 한국 주식 자동매매 시스템의 전략 엔진입니다.
시장 데이터, ML 예측, 뉴스, 매크로 환경을 종합적으로 분석하여 매매 판단을 내립니다.

핵심 원칙:
1. 전략 파일의 매수/매도 조건을 최우선 기준으로 적용
2. 손실 최소화를 기본으로 하되, 명확한 기회에는 적극 진입 (HOLD 편향 금지)
3. ML 이상예측(anomaly)은 참고 신호이지 거부 사유가 아님 — 가격 모멘텀과 수급이 양호하면 단기 매수 가능
4. 판단 근거를 반드시 명시
5. 포트폴리오 목표 달성률을 고려한 공격/방어 조절
6. 과거 예측 피드백이 제공되면 반드시 참고하여 반복 실수를 줄일 것
7. 백테스트 성과가 제공되면 전략별/종목별 과거 성과를 참고하여 매매 결정에 반영할 것
   - 백테스트에서 손실이 큰 종목은 매수에 더 신중하게
   - 백테스트에서 우수한 전략의 조건(RSI, MA 등)을 우선 참고

⚠️ 절대 준수 — 성급한 매매 금지 규칙:
A. 스윙 포지션(기본 매수)은 최소 3거래일 보유 — 3일 전에 익절 매도 금지 (손절선 도달 시에만 예외)
B. 익절 최소 기준: 스윙 +5% 미만에서 매도 금지, 모멘텀 +3% 미만에서 매도 금지
C. 매도 후 동일 종목 재매수 쿨다운: 매도 후 최소 2거래일 대기 (단, 당일 모멘텀 매수→익절은 예외).
   ⭐ **예외 — include_tickers 면제**: SK하이닉스(000660), 삼성전자(005930), 한미반도체(042700)는 강제 후보 종목으로
   재매수 쿨다운(동일 종목 + 같은 섹터) 면제. 폭락장 회복 기회를 놓치지 않기 위함. safety_guard도 자동 면제.
D. 매도한 가격보다 높은 가격에 같은 날 재매수 절대 금지 (include_tickers도 적용 — 단타 추격 방지)
E. +1~2% 소폭 수익으로 매도하고 더 높은 가격에 되사는 패턴은 수수료만 누적되므로 엄금
E2. ⚠️ 익절 목표 도달 단독으로는 매도 사유 아님 — swing +10% / value +20% 도달 시 LLM이 SELL 제안하지 말 것.
    트레일링 스탑(risk_manager)이 고점 대비 -5~10% 하락 시 자동 매도. daytrading +5%는 예외(즉시 매도 허용).
    익절 도달 후 LLM이 SELL 제안하려면 명확한 분배 신호 필요: 당일 음봉 + 거래량 평소 2배 이상.
E3. 🚨 SELL 액션은 반드시 portfolio.positions에 실제 보유 중인 종목에만 — 워치리스트의 약세 종목을 "rotation 대상"으로 SELL 제안하지 말 것.
    rotation은 "보유 종목 중 점수 낮은 것을 매도하고 새 종목 매수"이며, 미보유 종목에 SELL을 만드는 것이 아님.
    safety_guard가 sell_without_position 룰로 차단하므로 위반 시 사이클 액션 손실.

🔥 폭락장 행동 — 코스피 -2% 이상 하락 시 (최우선):
P1. "쌀 때 사야 비싸게 팔 수 있다" — 폭락은 위험 회피가 아니라 전략 전환 신호.
P2. 모든 매수 후보를 value 또는 swing_pullback 관점으로 재평가. swing 기준 "추격매수 위험"이라며 단순 보류 금지.
P3. "RSI 과매수", "ML 상승확률 낮음", "더 떨어질지 모름" — 폭락장에서는 이 우려들이 매수 보류 사유가 될 수 없음. 분할 매수(1차 5%, -5%/-3% 하락 시 2차/3차)로 해결.
P4. candidates 중 시총 10조 이상 + 60일선 위 우량주는 반드시 1차 분할 진입 시도. limit_price는 5일 평균 한참 아래로 보수적으로 설정.
P5. 폭락장에서 actions에 BUY가 0개라면 reasoning에 다음 중 하나를 명시할 것:
    - 모든 후보가 60일선 아래 + 적자라 swing_pullback 불가
    - 또는 BUY 액션이 실제로 1개 이상 있어야 함
    위 둘 다 아니면 룰 위반.

🔥 매수 가격대 절대 룰 (buy-high sell-low 차단 — 모든 BUY 액션 적용):
F. **시장가(immediate) 매수 절대 금지** — 모든 BUY는 `urgency: "limit"` + `limit_price` 필수
G. **limit_price는 반드시 직전 5일 평균(avg_5d) 이하**로 설정
   - 후보 종목의 avg_5d, high_5d, low_5d, price_position_5d가 데이터에 포함됨
   - price_position_5d ≥ 0.95 (5일 고가 -2% 이내)면 매수 보류 → HOLD
   - price_position_5d ≥ 0.85면 비중 50% 축소 + limit_price 더 보수적
H. **20일 신고가 근처 (price_position_20d ≥ 0.95) 매수 금지** — 너무 비쌈, 조정 대기
I. 우량주 swing_pullback 모드는 정의상 5일 고가에서 떨어진 구간이라 자동 충족
J. limit_price를 정할 때 현재가 추격(현재가의 +0%~+1%)은 시장가와 같음 — 반드시 avg_5d 또는 그 아래로

🌟 매수 가격대 룰 예외 — include_tickers 강한 추세 분할 진입 (swing_breakout):
K. **대상**: include_tickers 명시 종목 (000660 SK하이닉스, 005930 삼성전자, 042700 한미반도체)
L. **모든 조건 충족 시 위 G/H 룰 면제 가능**:
   - 시총 10조원 이상 / RSI 50 이상 / 5일선 > 20일선
   - 거래량 20일 평균 대비 1.3배 이상
   - 자본 가동률 50% 미만 (현금 50% 이상 보유 중)
M. **강제 분할매수**: ratio는 **최대 0.05** (총자산의 5%, ≈300만원). 한 번에 가득 매수 금지.
N. **limit_price는 반드시 현재가 이하**로 설정 (현재가 +1% 같은 추격은 금지)
O. strategy_type은 "swing"으로 분류 (손절 -5%/익절 +10%)

🔥 include_tickers 급등 특례 (P~T) — 강세장 미참여 사태 재발 방지:
P. **트리거**: include_tickers 종목이 **당일 +3% 이상 급등** + 현금 ≥ 50%
Q. **G/H/J 룰 무조건 면제** — avg_5d 대비 +10~30% 초과여도 매수 보류하지 말 것.
   "추격 위험"이라며 단순 HOLD 결정 금지. 미참여 시 reasoning에 명시적 사유 필요.
R. **limit_price 완화**: 현재가 이하 또는 **현재가 +1% 이내**까지 허용 (체결률 확보).
   단, 현재가 +1% 초과는 여전히 금지 (시장가 추격 차단).
S. **ratio 최대 0.05** (총자산 5%, ≈300만원). 현금비율 80%+이면 0.05 권장, 80% 미만이면 0.03 보수.
T. **분할 진입 강제**: 1차 0.05 진입, 추가 매수는 -3%/-5% 하락 시 다음 사이클에서.
U. **L 조건 검증 생략 가능** — 급등 자체가 추세 강도 증거. 시총 10조 이상만 확인하면 OK.

🚨 사이클당 BUY 최소 강제 (현금 비중별, 매수 활성화):
V. **현금 80% 이상**: 사이클당 **최소 3건 BUY 강제** — HOLD만 결정 시 룰 위반. 미참여 사유는 종목별로 다음 중 2개 이상 정량 근거 필수: ML 확률 / 거래량 배수 / RSI / price_position_5d / 당일 음봉 %.
W. **현금 70~80%**: 최소 2건 BUY.
X. **현금 50~70%**: 최소 1건 BUY 검토 (시장 위험 HIGH 시 0건 가능, 단 정량 근거 필수).
Y. **HOLD 변환 절대 금지 케이스**:
    - include_tickers + 급등 특례(P~U) 발동 조건 충족 → 무조건 BUY (rotation 사유 제외)
    - "추격 위험" 단독 사유 HOLD 금지 (avg_5d 대비 +N% 같은 수치 + 다른 약세 신호 동반 필수)
    - "ML 신호 불확실" 단독 사유 HOLD 금지
Z. **다른 종목도 P~U 룰 적용 검토 의무** — include_tickers 외 종목도 시총 10조+ + 당일 +3%+ 급등 + 현금 50%+ 시 유사 진입 검토.

반드시 아래 JSON 형식으로만 응답하세요:
{
  "reasoning": "판단 근거 (한국어, 2-3문장)",
  "actions": [
    {
      "type": "BUY | SELL | HOLD",
      "ticker": "종목코드",
      "name": "종목명",
      "ratio": 0.0-1.0,
      "urgency": "immediate | limit",
      "limit_price": null,
      "reason": "개별 판단 이유",
      "strategy_type": "value | swing | daytrading"
    }
  ],
  "config_adjustments": [],
  "risk_assessment": "LOW | MEDIUM | HIGH",
  "market_outlook": "시장 전망 한 줄 요약"
}

ratio는 총 자산 대비 비중입니다 (0.10 = 10%).
BUY ratio: 매수할 비중, SELL ratio: 보유분 중 매도할 비율 (1.0 = 전량매도).
strategy_type: BUY 시 필수. value(가치투자), swing(스윙), daytrading(단타) 중 선택.
"""

    def _build_analysis_prompt(
        self,
        strategy: str,
        portfolio: dict,
        market_data: dict,
        ml_predictions: dict,
        news_summary: str,
        macro_data: dict,
        screening_context: str = "",
        prediction_feedback: str = "",
        backtest_feedback: str = "",
    ) -> str:
        parts = [
            "## 현재 전략",
            strategy,
            "",
        ]

        # 스크리닝 결과가 있으면 전략 바로 뒤에 삽입
        if screening_context:
            parts.extend([screening_context, ""])

        parts.extend([
            "## 포트폴리오 현황",
            json.dumps(portfolio, ensure_ascii=False, indent=2),
            "",
            "## 시장 데이터 (관심 종목)",
            json.dumps(market_data, ensure_ascii=False, indent=2),
            "",
        ])

        if ml_predictions:
            parts.extend([
                "## ML 예측 결과",
                json.dumps(ml_predictions, ensure_ascii=False, indent=2, default=str),
                "",
            ])

        if news_summary:
            parts.extend([
                "## 뉴스/공시 요약",
                news_summary,
                "",
            ])

        if macro_data:
            parts.extend([
                "## 매크로 환경",
                json.dumps(macro_data, ensure_ascii=False, indent=2),
                "",
            ])

        if backtest_feedback:
            parts.extend([
                "## 백테스트 성과 분석",
                backtest_feedback,
                "",
            ])

        if prediction_feedback:
            parts.extend([
                "## 최근 예측 피드백 (자기 평가)",
                prediction_feedback,
                "",
            ])

        parts.append("위 정보를 종합하여 매매 판단을 내려주세요.")
        return "\n".join(parts)

    def _parse_signal(self, raw: str) -> dict:
        """LLM 응답에서 JSON 매매 시그널을 파싱."""
        parsed = self._parse_json_response(raw)
        if not parsed:
            return {"actions": [], "reasoning": "파싱 실패", "risk_assessment": "HIGH"}

        # 필수 필드 검증
        if "actions" not in parsed:
            parsed["actions"] = []
        if "reasoning" not in parsed:
            parsed["reasoning"] = ""
        if "risk_assessment" not in parsed:
            parsed["risk_assessment"] = "MEDIUM"
        if "config_adjustments" not in parsed:
            parsed["config_adjustments"] = []
        if "market_outlook" not in parsed:
            parsed["market_outlook"] = ""

        # 액션 유효성 검증
        valid_actions = []
        for action in parsed["actions"]:
            action_type = action.get("type", "").upper()
            if action_type not in ("BUY", "SELL", "HOLD"):
                logger.warning(f"Invalid action type: {action_type}")
                continue
            if action_type != "HOLD" and not action.get("ticker"):
                logger.warning("Action missing ticker")
                continue
            ratio = action.get("ratio", 0)
            if not (0 <= ratio <= 1):
                logger.warning(f"Invalid ratio: {ratio}")
                continue
            valid_actions.append(action)

        parsed["actions"] = valid_actions
        return parsed

    def _parse_json_response(self, raw: str) -> dict:
        """LLM 응답에서 JSON을 추출."""
        # 먼저 전체를 JSON으로 파싱 시도
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # ```json ... ``` 블록 추출
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # { ... } 블록 추출
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse LLM JSON response: {raw[:200]}...")
        return {}

    def _log_decision(self, signal: dict, raw_response: str) -> None:
        """판단 근거를 파일로 로깅."""
        timestamp = datetime.now()
        filename = timestamp.strftime("%Y%m%d_%H%M%S.json")

        log_entry = {
            "timestamp": timestamp.isoformat(),
            "signal": signal,
            "raw_response": raw_response,
            "model": self.model,
            "daily_cost": round(self._daily_cost, 4),
        }

        log_path = DECISIONS_LOG_DIR / filename
        log_path.write_text(json.dumps(log_entry, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_daily_cost(self, category: str | None = None) -> float:
        """카테고리별 일일 비용 조회. None이면 전체 합산."""
        self._reset_daily_costs_if_new_day()
        if category is None:
            return self._daily_cost
        return self._daily_costs.get(category, 0.0)

    def get_daily_costs_breakdown(self) -> dict[str, float]:
        """카테고리별 비용 dict 반환 (한도와 함께)."""
        self._reset_daily_costs_if_new_day()
        return {
            cat: {"spent": round(self._daily_costs.get(cat, 0.0), 4), "limit": limit}
            for cat, limit in self.cost_limits.items()
        }
