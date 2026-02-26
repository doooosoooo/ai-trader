# AI Trader — 한국 주식 AI 자동매매 시스템

LLM(Claude)을 전략 두뇌로, ML 모델을 예측 보조로 사용하는 한국 주식 자동매매 시스템.

## 빠른 시작

```bash
# 1. 환경 설정
cd ~/ai-trader
cp config/.env.example config/.env
# config/.env에 API 키 입력

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 시뮬레이션 모드로 시작
python main.py

# 4. PM2로 상시 실행
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

## CLI 사용법

```bash
python -m interfaces.cli status          # 시스템 상태
python -m interfaces.cli portfolio       # 포트폴리오
python -m interfaces.cli history         # 매매 이력
python -m interfaces.cli chat "단타로 전환해"  # 자연어 전략 변경
python -m interfaces.cli strategy show   # 현재 전략
python -m interfaces.cli ml train        # ML 학습
python -m interfaces.cli review today    # 일일 복기
```

## 필요한 API 키

| 키 | 용도 |
|---|---|
| KIS_APP_KEY, KIS_APP_SECRET | 한국투자증권 API |
| KIS_ACCOUNT_NO | 증권 계좌번호 |
| ANTHROPIC_API_KEY | Claude API |
| TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID | Telegram 봇 |
| DART_API_KEY | DART 공시 API |
