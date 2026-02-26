# AI Trader — Claude Code 에이전트 컨텍스트

## 프로젝트 개요
한국 주식 AI 자동매매 시스템. LLM(Claude)을 전략 두뇌로, ML 모델을 예측 보조로 사용.

## 핵심 규칙

### 절대 금지
- **PM2를 함부로 재시작하지 마라** — 장중에 `pm2 restart`하면 실시간 매매에 영향
- **config/.env를 수정하지 마라** — API 키가 들어있음
- **safety-rules.yaml을 수정하지 마라** — 하드코딩된 안전장치
- **live 모드에서 테스트하지 마라** — 실제 돈이 관련됨
- **데이터베이스(trader.db)를 직접 수정하지 마라** — 포트폴리오/매매이력 소실 위험

### 작업 전 확인
- `pm2 status ai-trader`로 시스템 실행 상태 확인
- 장중(09:00~15:30)에는 코드 수정 후 즉시 반영하지 않기
- config 변경 시 `config/settings.yaml` 또는 `config/trading-params.yaml`만 수정

## 프로젝트 구조
```
ai-trader/
├── config/          — 설정 (settings.yaml, safety-rules.yaml, trading-params.yaml)
├── core/            — 핵심 모듈 (LLM, KIS API, Safety Guard, 포트폴리오, 주문 실행)
├── data/            — 데이터 파이프라인 (수집기, 피처 엔지니어링)
├── ml/              — ML 모델 (학습, 추론)
├── interfaces/      — CLI (Typer), Telegram 봇
├── review/          — 복기 시스템
├── simulation/      — 시뮬레이션/백테스팅
├── strategies/      — 전략 프롬프트 (active.md가 현재 활성 전략)
├── logs/            — 로그 (판단 근거, 매매, 에러, 복기)
├── main.py          — 메인 엔트리포인트
└── scheduler.py     — APScheduler 스케줄러
```

## 기술 스택
- Python 3.12, Claude API (Sonnet), KIS OpenAPI
- SQLite, APScheduler, LightGBM, Typer, python-telegram-bot

## 실행
```bash
source .venv/bin/activate
pm2 start ecosystem.config.js        # 프로덕션
python main.py                        # 직접 실행
python -m interfaces.cli status       # CLI
```

## 모드
- `simulation`: 가상 매매 (기본값, 안전)
- `live`: 실전 매매 (주의!)

## 테스트
코드 수정 후:
1. `python -c "from core.config_manager import ConfigManager; c = ConfigManager(); print(c.settings)"`
2. 시뮬레이션 모드에서 1사이클 수동 테스트
3. 이상 없으면 PM2 재시작 (장 마감 후에만)
