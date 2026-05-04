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
        self.cost_limit_daily = llm_config.get("cost_limit_daily_usd", 10.0)
        self._daily_cost = 0.0
        self._cost_date = ""
        DECISIONS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _check_cost_limit(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._cost_date != today:
            self._daily_cost = 0.0
            self._cost_date = today
        if self._daily_cost >= self.cost_limit_daily:
            logger.warning(f"LLM daily cost limit reached: ${self._daily_cost:.2f}")
            return False
        return True

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

    def _track_cost(self, usage: dict, model: str) -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        in_price, out_price = self._resolve_pricing(model)
        cost = (input_tokens * in_price / 1_000_000) + (output_tokens * out_price / 1_000_000)
        self._daily_cost += cost

    def _call_llm(self, system_prompt: str, user_message: str, model: str | None = None) -> str:
        if not self._check_cost_limit():
            raise RuntimeError("LLM daily cost limit reached")

        use_model = model or self.model
        # Opus 4.7부터 temperature 파라미터 deprecated (400 에러). SDK default 사용.
        response = self.client.messages.create(
            model=use_model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        self._track_cost({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }, model=use_model)

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
- 익절 목표 도달 (value +20%, swing +10%, daytrading +5%) — 보유일 무관 즉시 매도
- 트레일링 스탑 (수익 구간에서 고점 대비 -5%)

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
C. 매도 후 동일 종목 재매수 쿨다운: 매도 후 최소 2거래일 대기 (단, 당일 모멘텀 매수→익절은 예외)
D. 매도한 가격보다 높은 가격에 같은 날 재매수 절대 금지
E. +1~2% 소폭 수익으로 매도하고 더 높은 가격에 되사는 패턴은 수수료만 누적되므로 엄금

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

    def get_daily_cost(self) -> float:
        return self._daily_cost
