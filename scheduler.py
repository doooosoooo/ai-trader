"""분석 사이클 스케줄러 — APScheduler 기반."""

import asyncio
from datetime import datetime, time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


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

        # 뉴스/공시 체크
        self.scheduler.add_job(
            self._safe_run(self.system.cycle_news_check),
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

        # 장 시작 전 초기화 (08:50)
        self.scheduler.add_job(
            self._safe_run(self.system.on_market_open),
            CronTrigger(hour=8, minute=50, day_of_week="mon-fri"),
            id="market_open",
            name="장 시작 초기화",
        )

        # 장 마감 후 정리 (15:40)
        self.scheduler.add_job(
            self._safe_run(self.system.on_market_close),
            CronTrigger(hour=15, minute=40, day_of_week="mon-fri"),
            id="market_close",
            name="장 마감 정리",
        )

        # 일일 복기 (16:00)
        self.scheduler.add_job(
            self._safe_run(self.system.run_daily_review),
            CronTrigger(hour=16, minute=0, day_of_week="mon-fri"),
            id="daily_review",
            name="일일 복기",
        )

        # ML 모델 재학습 (17:00, 배치 모드인 경우)
        ml_config = self.config.get("ml", {})
        if ml_config.get("training_mode") == "batch":
            if ml_config.get("retrain_schedule") == "daily":
                self.scheduler.add_job(
                    self._safe_run(self.system.train_ml_models),
                    CronTrigger(hour=17, minute=0, day_of_week="mon-fri"),
                    id="ml_retrain",
                    name="ML 재학습",
                )

        # 주간 리포트 (토요일 10:00)
        self.scheduler.add_job(
            self._safe_run(self.system.run_weekly_review),
            CronTrigger(hour=10, minute=0, day_of_week="sat"),
            id="weekly_review",
            name="주간 리포트",
        )

    def _safe_run(self, func):
        """에러가 발생해도 스케줄러가 멈추지 않도록 래핑."""
        async def wrapper():
            if not self._is_trading_hours() and "circuit" not in str(func):
                return

            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception as e:
                logger.error(f"Scheduled job failed [{func.__name__}]: {e}")

        return wrapper

    def _is_trading_hours(self) -> bool:
        """현재 장 시간인지 확인."""
        now = datetime.now()
        # 주말 제외
        if now.weekday() >= 5:
            return False
        # 09:00 ~ 15:35
        market_open = time(9, 0)
        market_close = time(15, 35)
        return market_open <= now.time() <= market_close

    def start(self):
        self.scheduler.start()
        logger.info("Trading scheduler started")

    def stop(self):
        self.scheduler.shutdown(wait=False)
        logger.info("Trading scheduler stopped")

    def pause_trading_jobs(self):
        """매매 관련 잡만 일시 중지."""
        for job_id in ("llm_analysis", "data_collection"):
            self.scheduler.pause_job(job_id)
        logger.info("Trading jobs paused")

    def resume_trading_jobs(self):
        for job_id in ("llm_analysis", "data_collection"):
            self.scheduler.resume_job(job_id)
        logger.info("Trading jobs resumed")
