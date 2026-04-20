# 전략 파라미터 정리 작업 (완료)

생성일: 2026-04-19
배경: 전략 문서(strategies/active.md)와 config/trading-params.yaml 불일치 조사 중 발견된 개선사항.

## 완료 (2026-04-19)

- [x] **버그 2 수정**: `hold_period_days` → `holding_period_days` 키 불일치
  - [core/risk_manager.py:86](../core/risk_manager.py#L86)
  - [main.py:395](../main.py#L395)
  - 영향: 이전엔 기본값 20일 조용히 적용 → 이제 설정값 14일 정상 적용
- [x] **버그 1 수정**: `trailing_stop_pct` 0.1 → 0.05
  - [config/trading-params.yaml:27](../config/trading-params.yaml#L27) 및 .example
  - 영향: 전략 문서(-5% 트레일링 스탑)와 일치
- [x] **작업 3**: main.py LLM 컨텍스트에서 `dist_to_stop_loss`를 전략 타입별(`pos.rules["stop_loss"]`)로 계산하도록 수정
- [x] **작업 4**: trading-params.yaml에서 `stop_loss_pct`, `take_profit_pct` dead 키 삭제 (.example 포함). 참조 정리: `interfaces/telegram_bot.py`(opt_apply, /param 예시), `core/llm_engine.py`(자연어 조정 프롬프트), `core/risk_manager.py`(dead 변수)
- [x] **작업 5**: 최종 yaml 상태 검증 — ConfigManager 로드 OK

## 참고 (이력 보존용)

### 작업 3: LLM 컨텍스트를 전략 타입별 손절값으로 수정

**문제**: [main.py:393-414](../main.py#L393-L414)가 전역 `stop_loss_pct`(-0.07)로 "손절까지 거리"를 계산해 LLM에 전달. 실제 손절은 `Position.STRATEGY_RULES`(value -10%, swing -5%, daytrading -3%)로 동작 → LLM이 swing 종목을 -7%에서 손절된다고 오인할 수 있음.

**수정 방법**: 라인 393의 `sl_pct = t_params.get("stop_loss_pct", -0.07)` 제거하고, 라인 405 `dist_to_sl` 계산을 포지션마다 다르게:
```python
pos_sl = pos.rules["stop_loss"]  # strategy_type별
dist_to_sl = pos.pnl_pct - pos_sl
```
라인 414 `pos_dict["trailing_stop_pct"]`도 현재 전역값 → 트레일링 스탑은 전역값이 맞으니 유지 OK.

### 작업 4: trading-params.yaml에서 dead 키 삭제

- `stop_loss_pct: -0.07` — Position.STRATEGY_RULES가 처리하므로 dead. 단 작업 3 완료 후에만 삭제 가능 (현재는 main.py에서 읽고 있음)
- `take_profit_pct: 0.2` — dead (Position.STRATEGY_RULES가 처리)
- 단, [interfaces/telegram_bot.py:1248-1251](../interfaces/telegram_bot.py#L1248-L1251), [core/llm_engine.py:260-263](../core/llm_engine.py#L260-L263)에서 이 키를 사용자/LLM이 동적 조정 대상으로 참조 → 삭제 시 함께 정리 필요
- `optimizer_main.py` 및 `run_backtest.py`는 별개 목적(백테스트/최적화)이므로 유지

### 작업 5: trading-params.yaml 최종 상태

```yaml
# 남기는 키 (실사용)
trailing_stop_pct: 0.05
holding_period_days: {min: 3, max: 14}  # max는 전역 안전장치
position_size_pct: 0.13
max_cash_ratio: 0.2
indicators: {...}
market_conditions: {...}
portfolio_targets: {...}
unfilled_cancel_threshold_pct: 0.01
unfilled_resubmit: true

# 삭제 대상 (작업 3 이후)
# stop_loss_pct: -0.07   ← DELETE
# take_profit_pct: 0.2   ← DELETE
```

## 주의사항

- 작업 순서: **3 → 4 → 5** 순서 엄수 (main.py가 여전히 `stop_loss_pct`를 읽음)
- 장중(평일 09:00~15:30) 수정 금지, 장 마감 후 PM2 재시작 필요
- 텔레그램 `/param` 명령 및 LLM 프롬프트 정책도 동시 정리 필요
