# AI Trader — 한국 주식 AI 자동매매 시스템

LLM(Claude)을 전략 두뇌로, ML 모델을 예측 보조로 사용하는 한국 주식 자동매매 시스템.

## 아키텍처

```
┌─────────────────────────────────────────────────┐
│                  PM2 프로세스                      │
│  ┌─────────────┐    ┌──────────────────┐         │
│  │  ai-trader  │    │  ai-optimizer    │         │
│  │  (main.py)  │    │ (optimizer_main) │         │
│  └──────┬──────┘    └────────┬─────────┘         │
│         │                    │                    │
│    ┌────┴────┐         ┌────┴────┐               │
│    │ 2단계   │         │ 백테스트 │               │
│    │ LLM분석 │         │ 최적화   │               │
│    └────┬────┘         └─────────┘               │
└─────────┼────────────────────────────────────────┘
          │
    ┌─────┴─────┐
    │ Stage 1   │ Sonnet 4.6 — 빠른 분류
    │ Stage 2   │ Opus 4.6 — 깊이 분석
    └───────────┘
```

### 2개 프로세스 구조
- **ai-trader** (`main.py`): 실시간 매매, 데이터 수집, LLM 분석, 주문 실행
- **ai-optimizer** (`optimizer_main.py`): 파라미터 최적화, 백테스트, 텔레그램 제안

### 2단계 LLM 분석
- **1단계 (Sonnet)**: 보유종목 빠른 분류 + 매수 후보 선별
- **2단계 (Opus)**: 후보 종목 깊이 분석 → BUY/SELL 결정

### 멀티전략 (strategy_type)
| 유형 | 손절 | 익절 | 최소보유 | 스크리닝탈락 매도 |
|------|------|------|---------|----------------|
| 💎 value | -10% | +20% | 10일 | 안 함 |
| 🔄 swing | -5% | +10% | 3일 | 즉시 |
| ⚡ daytrading | -3% | +5% | 0일 | 즉시 |

## 빠른 시작

```bash
# 1. 환경 설정
cd ~/ai-trader
cp config/.env.example config/.env
# config/.env에 API 키 입력

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. PM2로 시작
pm2 start start_trader.sh --name ai-trader --cwd /home/ubuntu/ai-trader
pm2 start start_optimizer.sh --name ai-optimizer --cwd /home/ubuntu/ai-trader
pm2 save
pm2 startup
```

## 프로젝트 구조

```
ai-trader/
├── config/                  — 설정 파일
│   ├── settings.yaml        — 시스템 설정
│   ├── safety-rules.yaml    — 안전장치 (LLM 변경 불가)
│   ├── trading-params.yaml  — 매매 파라미터 (LLM 조정 가능)
│   ├── screening-params.yaml— 스크리닝 설정
│   └── optimizer-params.yaml— 최적화 설정
├── core/                    — 핵심 모듈
│   ├── llm_engine.py        — 2단계 LLM 분석 (Sonnet + Opus)
│   ├── executor.py          — 주문 실행 (시초가 차단, 분봉 확인)
│   ├── risk_manager.py      — 리스크 관리 (유형별 손절, 갭상승 익절)
│   ├── portfolio.py         — 포트폴리오 (strategy_type 지원)
│   ├── circuit_breaker.py   — 서킷브레이커 (HALTED -4%, 해제 -3%)
│   ├── market_data.py       — KIS API 연동
│   ├── account_manager.py   — 계좌 동기화
│   ├── notification.py      — 텔레그램 알림 (유형별 그룹)
│   └── safety_guard.py      — 안전 검증
├── data/                    — 데이터 파이프라인
│   ├── collectors/          — 수집기 (가격, 분봉, 뉴스, 스크리닝)
│   └── storage/             — SQLite DB (일봉, 분봉, 포트폴리오)
├── ml/                      — ML 모델 (LightGBM)
├── simulation/              — 백테스트 엔진
│   ├── backtest.py          — 백테스터 (분봉 지원, 스크리닝 시뮬레이션)
│   ├── strategies.py        — 전략 프리셋
│   └── optimizer.py         — 그리드 서치, Walk-Forward
├── interfaces/              — 사용자 인터페이스
│   └── telegram_bot.py      — 텔레그램 봇 (명령, 알림, 승인)
├── strategies/
│   └── active.md            — 현재 활성 전략 (v4 멀티전략)
├── main.py                  — ai-trader 엔트리포인트
├── optimizer_main.py        — ai-optimizer 엔트리포인트
├── start_trader.sh          — PM2 실행 래퍼
└── start_optimizer.sh       — PM2 실행 래퍼
```

## 텔레그램 명령어

| 명령 | 설명 |
|------|------|
| /status | 시스템 상태 + 보유종목 (유형별 그룹) |
| /portfolio | 포트폴리오 상세 |
| /today | 오늘 매매 내역 |
| /analysis | 수동 LLM 분석 실행 |
| /screen | 수동 스크리닝 실행 |
| /orders | 미체결 주문 현황 |
| /trades | 최근 매매 이력 |
| /param | 파라미터 조회/변경 |
| /setcapital | 초기자본 설정 (입출금 반영) |
| /backtest | 백테스트 실행 |

## AI Optimizer

별도 프로세스로 30분마다 파라미터 최적화를 실행합니다.
- 1,400+ 종목의 5년치 일봉 데이터 기반 백테스트
- 개선된 파라미터 발견 시 텔레그램으로 알림 (적용/무시 버튼)
- 장중: daily 그리드 (108 조합) / 장외: deep 그리드 (1,920 조합)

## 핵심 안전장치

- **서킷브레이커**: 코스피 -4% → HALTED, -3% 회복 시 해제 (30분 쿨다운)
- **하드 손절**: strategy_type별 다른 손절선 (코드 레벨 강제)
- **시초가 30분 매수 금지**: 변동성 함정 회피
- **5분봉 양봉 확인**: 떨어지는 칼날 차단
- **PER/PBR 자동 분류**: LLM 분류 보정
- **매도 원칙**: 사실 기반만 (예측 기반 매도 금지)

## 필요한 API 키

| 키 | 용도 |
|---|---|
| KIS_APP_KEY, KIS_APP_SECRET | 한국투자증권 API |
| KIS_ACCOUNT_NO | 증권 계좌번호 |
| ANTHROPIC_API_KEY | Claude API (Sonnet + Opus) |
| TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID | Telegram 봇 |
| DART_API_KEY | DART 공시 API |

## 기술 스택

- Python 3.12, Claude API (Sonnet 4.6 + Opus 4.6), KIS OpenAPI
- SQLite, APScheduler, LightGBM, python-telegram-bot
- PM2 (프로세스 관리), Naver Finance API (스크리닝)
