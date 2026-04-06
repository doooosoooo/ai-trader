"""계좌 동기화 + 모드 전환."""

from loguru import logger

from core.analysis_store import TICKER_NAMES
from core.portfolio import Position


class AccountManager:
    """브로커 계좌 동기화 및 운영 모드 전환."""

    def __init__(self, market_client, portfolio, config_manager, executor):
        """
        Args:
            market_client: KISClient 인스턴스
            portfolio: Portfolio 인스턴스
            config_manager: ConfigManager 인스턴스
            executor: OrderExecutor 인스턴스
        """
        self.market_client = market_client
        self.portfolio = portfolio
        self.config_manager = config_manager
        self.executor = executor

    def sync_account_from_broker(self) -> dict | None:
        """KIS API에서 실제 계좌 잔고/보유종목을 가져와 포트폴리오 동기화.

        Returns:
            동기화 결과 dict 또는 실패 시 None
        """
        try:
            account = self.market_client.get_account_balance()
            # API 실패로 빈 결과가 돌아온 경우 기존 포트폴리오 유지
            if account["total_asset"] == 0 and not account["positions"]:
                raise RuntimeError("Empty account data returned (API may be unavailable)")
            logger.info(f"Account sync: cash={account['cash']:,}, "
                       f"positions={len(account['positions'])}, "
                       f"total={account['total_asset']:,}")

            # 포트폴리오 현금 동기화
            self.portfolio.cash = account["cash"]
            # initial_capital은 최초 1회만 설정 (이후 동기화에서 덮어쓰면 PnL 추적이 깨짐)
            if self.portfolio.initial_capital == 0 and account["total_asset"] > 0:
                self.portfolio.initial_capital = account["total_asset"]

            # 보유종목 동기화 — 기존 포지션의 peak_price, bought_at 보존
            old_positions = {t: p for t, p in self.portfolio.positions.items()}
            self.portfolio.positions.clear()
            for pos_data in account["positions"]:
                ticker = pos_data["ticker"]
                cur_price = pos_data["current_price"]
                old_pos = old_positions.get(ticker)
                # 기존 peak_price, bought_at 유지
                prev_peak = old_pos.peak_price if old_pos else cur_price
                prev_bought = old_pos.bought_at if old_pos else ""
                prev_strategy = old_pos.strategy_type if old_pos else "swing"
                self.portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    name=pos_data["name"],
                    quantity=pos_data["quantity"],
                    avg_price=pos_data["avg_price"],
                    current_price=cur_price,
                    peak_price=max(prev_peak, cur_price),
                    bought_at=prev_bought,
                    strategy_type=prev_strategy,
                )
                # TICKER_NAMES에 추가
                if pos_data["ticker"] not in TICKER_NAMES and pos_data["name"]:
                    TICKER_NAMES[pos_data["ticker"]] = pos_data["name"]

            # HWM 갱신
            if self.portfolio.total_asset > self.portfolio.high_water_mark:
                self.portfolio.high_water_mark = self.portfolio.total_asset
            self.portfolio._save_state()
            logger.info(f"Portfolio synced from broker: {len(account['positions'])} positions")
            return account

        except Exception as e:
            logger.error(f"Account sync failed: {e}")
            raise

    def switch_mode(self, mode: str) -> str:
        """모드 전환 + 필요 시 계좌 동기화.

        Returns:
            결과 메시지
        """
        old_mode = self.config_manager.get_mode()

        self.config_manager.set_mode(mode)
        self.portfolio.mode = mode
        self.executor.mode = mode

        if mode == "live":
            # 실계좌 동기화
            try:
                account = self.sync_account_from_broker()
                pos_count = len(account["positions"])
                msg = (
                    f"🔴 LIVE 모드 전환 완료\n"
                    f"계좌 동기화 성공\n"
                    f"  예수금: {account['cash']:,.0f}원\n"
                    f"  보유종목: {pos_count}개\n"
                    f"  총평가: {account['total_asset']:,.0f}원"
                )
            except Exception as e:
                logger.error(f"Mode switch account sync failed: {e}")
                msg = (
                    f"🔴 LIVE 모드 전환 완료\n"
                    f"⚠️ 계좌 동기화 실패 — 수동 확인 필요"
                )
        else:
            msg = f"🔵 시뮬레이션 모드 전환 완료 (이전: {old_mode})"

        logger.info(f"Mode switched: {old_mode} → {mode}")
        return msg
