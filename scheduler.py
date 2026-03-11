"""분석 사이클 스케줄러 — APScheduler 기반."""

import asyncio
from datetime import datetime, time

import holidays
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

# 한국 공휴일 (매년 자동 갱신)
_kr_holidays = holidays.KR(years=range(2025, 2030))


class TradingScheduler:
    """매매 시스템 스케줄러."""

    def __init__(self, system, config: dict):
        self.system = system
        self.config = config
        self.scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
        self._setup_jobs()

    def _setup_jobs(self):
        schedule = self.config.get("schedule", {})

        data_interval = schedule.get("data_collection_interval", 5)
        llm_interval = schedule.get("llm_analysis_interval", 15)
        news_interval = schedule.get("news_check_interval", 30)

        # 장중 데이터 수집 (09:00 ~ 15:35, 매 N분)
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_data_collection),
            IntervalTrigger(minutes=data_interval),
            id="data_collection",
            name="데이터 수집",
            misfire_grace_time=60,
        )

        # 장중 LLM 분석 (09:05 ~ 15:25, 매 M분)
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_llm_analysis),
            IntervalTrigger(minutes=llm_interval),
            id="llm_analysis",
            name="LLM 분석",
            misfire_grace_time=120,
        )

        # 뉴스/공시 체크 — 주말/공휴일 포함 상시 수집
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_news_check, check_hours=False, check_trading_day=False),
            IntervalTrigger(minutes=news_interval),
            id="news_check",
            name="뉴스 체크",
            misfire_grace_time=60,
        )

        # 서킷브레이커 체크 (매 1분)
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_circuit_check),
            IntervalTrigger(minutes=1),
            id="circuit_check",
            name="서킷브레이커 체크",
            misfire_grace_time=30,
        )

        # 미체결 주문 관리 (매 2분)
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_unfilled_order_check),
            IntervalTrigger(minutes=2),
            id="unfilled_order_check",
            name="미체결 주문 관리",
            misfire_grace_time=30,
        )

        # 장마감 전 미체결 일괄 취소 (15:22) — 안전망
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_unfilled_order_check, check_hours=False),
            CronTrigger(hour=15, minute=22, day_of_week="mon-fri"),
            id="eod_unfilled_cancel",
            name="장마감 미체결 취소",
        )

        # 장전 종목 스크리닝 (08:45) — on_market_open 전에 완료
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_screening, check_hours=False),
            CronTrigger(hour=8, minute=45, day_of_week="mon-fri"),
            id="daily_screening",
            name="종목 스크리닝",
        )

        # 장 시작 전 초기화 (08:50) — 장외시간이므로 check_hours=False
        self.scheduler.add_job(
            self._safe_run(self.system.on_market_open, check_hours=False),
            CronTrigger(hour=8, minute=50, day_of_week="mon-fri"),
            id="market_open",
            name="장 시작 초기화",
        )

        # 장 마감 후 정리 (15:40) — 장외시간이므로 check_hours=False
        self.scheduler.add_job(
            self._safe_run(self.system.on_market_close, check_hours=False),
            CronTrigger(hour=15, minute=40, day_of_week="mon-fri"),
            id="market_close",
            name="장 마감 정리",
        )

        # 일일 복기 (16:00) — 장외시간이므로 check_hours=False
        self.scheduler.add_job(
            self._safe_run(self.system.run_daily_review, check_hours=False),
            CronTrigger(hour=16, minute=0, day_of_week="mon-fri"),
            id="daily_review",
            name="일일 복기",
        )

        # ML 모델 재학습 (17:00, 배치 모드인 경우)
        ml_config = self.config.get("ml", {})
        if ml_config.get("training_mode") == "batch":
            if ml_config.get("retrain_schedule") == "daily":
                self.scheduler.add_job(
                    self._safe_run(self.system.train_ml_models, check_hours=False),
                    CronTrigger(hour=17, minute=0, day_of_week="mon-fri"),
                    id="ml_retrain",
                    name="ML 재학습",
                )

        # 주간 리포트 (토요일 10:00) — 주말이므로 거래일 체크도 skip
        self.scheduler.add_job(
            self._safe_run(self.system.run_weekly_review, check_hours=False, check_trading_day=False),
            CronTrigger(hour=10, minute=0, day_of_week="sat"),
            id="weekly_review",
            name="주간 리포트",
        )

        # 일일 백테스트 (매일 18:00) — 장 마감 후 자동 실행
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_backtest, check_hours=False, check_trading_day=False),
            CronTrigger(hour=18, minute=0),
            id="daily_backtest",
            name="일일 백테스트",
        )

        # 장외 LLM 분석 (매 2시간, 장중 제외) — 최신 데이터 기반 분석 유지
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_llm_analysis, check_hours=False, check_trading_day=False),
            CronTrigger(hour="0,2,4,6,8,16,18,20,22", minute=30),
            id="offhours_analysis",
            name="장외 LLM 분석",
        )

    def _safe_run(self, func, check_hours=True, check_trading_day=True):
        """에러가 발생해도 스케줄러가 멈추지 않도록 래핑."""
        func_name = getattr(func, '__name__', str(func))

        async def wrapper():
            logger.info(f"Job triggered: {func_name}")
            if check_trading_day and not self._is_trading_day():
                logger.info(f"Not a trading day (holiday/weekend), skipping {func_name}")
                return
            if check_hours and not self._is_trading_hours():
                logger.info(f"Outside trading hours, skipping {func_name}")
                return

            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    await asyncio.get_event_loop().run_in_executor(None, func)
            except Exception as e:
                logger.error(f"Scheduled job failed [{func_name}]: {e}")

        return wrapper

    def _is_trading_day(self) -> bool:
        """오늘이 거래일(평일 + 공휴일 아님)인지 확인."""
        today = datetime.now().date()
        if today.weekday() >= 5:
            return False
        if today in _kr_holidays:
            return False
        return True

    def _is_trading_hours(self) -> bool:
        """현재 장 시간인지 확인."""
        if not self._is_trading_day():
            return False
        # 09:00 ~ 15:35
        now = datetime.now()
        market_open = time(9, 0)
        market_close = time(15, 35)
        return market_open <= now.time() <= market_close

    def start(self):
        self.scheduler.start()
        logger.info("Trading scheduler started")

    def stop(self):
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
                logger.info("Trading scheduler stopped")
        except Exception:
            pass

    def pause_trading_jobs(self):
        """매매 관련 잡만 일시 중지."""
        for job_id in ("llm_analysis", "data_collection", "unfilled_order_check"):
            self.scheduler.pause_job(job_id)
        logger.info("Trading jobs paused")

    def resume_trading_jobs(self):
        for job_id in ("llm_analysis", "data_collection", "unfilled_order_check"):
            self.scheduler.resume_job(job_id)
        logger.info("Trading jobs resumed")
