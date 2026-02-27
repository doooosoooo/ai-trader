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
        self.max_tokens = llm_config.get("max_tokens", 2048)
        self.temperature = llm_config.get("temperature", 0.3)
        self.cost_limit_daily = llm_config.get("cost_limit_daily_usd", 5.0)
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

    def _track_cost(self, usage: dict) -> None:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        # Sonnet 가격 기준 (대략적)
        cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
        self._daily_cost += cost

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        if not self._check_cost_limit():
            raise RuntimeError("LLM daily cost limit reached")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        self._track_cost({
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        })

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
    ) -> dict:
        """시장 분석 및 매매 판단.

        Returns:
            파싱된 매매 시그널 dict
        """
        strategy = self.load_strategy()

        system_prompt = self._build_system_prompt()
        user_message = self._build_analysis_prompt(
            strategy=strategy,
            portfolio=portfolio,
            market_data=market_data,
            ml_predictions=ml_predictions or {},
            news_summary=news_summary,
            macro_data=macro_data or {},
            screening_context=screening_context,
        )

        raw_response = self._call_llm(system_prompt, user_message)
        signal = self._parse_signal(raw_response)

        # 판단 근거 로깅
        self._log_decision(signal, raw_response)

        return signal

    def interpret_natural_language(self, user_input: str, current_params: dict) -> dict:
        """자연어 전략 변경 → config 조정 목록 반환."""
        system_prompt = """당신은 AI 트레이딩 시스템의 설정 관리자입니다.
사용자의 자연어 지시를 해석하여 매매 파라미터를 조정합니다.

현재 조정 가능한 파라미터:
- holding_period_days.min / holding_period_days.max: 보유 기간
- take_profit_pct: 익절 비율 (0.02~0.50)
- stop_loss_pct: 손절 비율 (-0.20~-0.01)
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
1. 보수적 판단 — 확신이 없으면 HOLD
2. 손실 최소화 > 수익 극대화
3. 판단 근거를 반드시 명시
4. 포트폴리오 목표 달성률을 고려한 공격/방어 조절

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
      "reason": "개별 판단 이유"
    }
  ],
  "config_adjustments": [],
  "risk_assessment": "LOW | MEDIUM | HIGH",
  "market_outlook": "시장 전망 한 줄 요약"
}

ratio는 총 자산 대비 비중입니다 (0.10 = 10%).
BUY ratio: 매수할 비중, SELL ratio: 보유분 중 매도할 비율 (1.0 = 전량매도).
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
                json.dumps(ml_predictions, ensure_ascii=False, indent=2),
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
