"""CLI 인터페이스 — Typer 기반."""

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

app = typer.Typer(name="ai-trader", help="AI 주식 자동매매 시스템")
console = Console()


def _get_system():
    """TradingSystem 인스턴스를 가져온다 (lazy)."""
    from main import create_system
    return create_system()


# --- 시스템 제어 ---

@app.command()
def start():
    """시스템 시작."""
    console.print("[bold green]AI Trader 시작...[/]")
    system = _get_system()
    system.run()


@app.command()
def stop():
    """시스템 중지."""
    console.print("[bold red]AI Trader 중지[/]")


@app.command()
def status():
    """현재 시스템 상태 조회."""
    system = _get_system()
    st = system.get_status()

    table = Table(title="시스템 상태")
    table.add_column("항목", style="cyan")
    table.add_column("값", style="white")

    table.add_row("모드", st.get("mode", "?"))
    table.add_row("서킷브레이커", st.get("circuit_state", "?"))
    table.add_row("총자산", f"{st.get('total_asset', 0):,.0f}원")
    table.add_row("현금", f"{st.get('cash', 0):,.0f}원")
    table.add_row("수익률", st.get("total_pnl_pct", "0%"))
    table.add_row("보유종목", str(st.get("num_positions", 0)))
    table.add_row("LLM 일일비용", f"${st.get('llm_daily_cost', 0):.2f}")

    console.print(table)


# --- 전략 ---

strategy_app = typer.Typer(help="전략 관리")
app.add_typer(strategy_app, name="strategy")


@strategy_app.command("show")
def strategy_show():
    """현재 전략 표시."""
    active = Path("strategies/active.md")
    if active.exists():
        content = active.read_text(encoding="utf-8")
        console.print(Panel(content, title="현재 전략", border_style="green"))
    else:
        console.print("[yellow]활성 전략 파일 없음[/]")


@strategy_app.command("list")
def strategy_list():
    """전략 버전 목록."""
    archive = Path("strategies/archive")
    if not archive.exists():
        console.print("[yellow]아카이브 없음[/]")
        return

    table = Table(title="전략 아카이브")
    table.add_column("파일명", style="cyan")
    table.add_column("크기", style="white")

    for f in sorted(archive.glob("*.md")):
        table.add_row(f.stem, f"{f.stat().st_size:,}B")

    console.print(table)


@strategy_app.command("rollback")
def strategy_rollback(version: str):
    """이전 버전으로 롤백."""
    import shutil
    archive_file = Path(f"strategies/archive/{version}.md")
    if not archive_file.exists():
        # 부분 매칭 시도
        matches = list(Path("strategies/archive").glob(f"*{version}*"))
        if matches:
            archive_file = matches[0]
        else:
            console.print(f"[red]전략 '{version}' 을(를) 찾을 수 없음[/]")
            return

    active = Path("strategies/active.md")
    if active.exists():
        # 현재 전략 백업
        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = Path(f"strategies/archive/backup_{timestamp}.md")
        shutil.copy2(active, backup)
        console.print(f"현재 전략 백업: {backup.name}")

    shutil.copy2(archive_file, active)
    console.print(f"[green]전략 롤백 완료: {archive_file.stem}[/]")


# --- 포트폴리오 ---

@app.command()
def portfolio():
    """보유 종목 + 수익률."""
    system = _get_system()
    summary = system.portfolio.get_summary()

    console.print(Panel(
        f"총자산: {summary['total_asset']:,.0f}원 | "
        f"현금: {summary['cash']:,.0f}원 ({summary['cash_ratio']}) | "
        f"수익률: {summary['total_pnl_pct']}",
        title="포트폴리오",
    ))

    if summary.get("positions"):
        table = Table()
        table.add_column("종목", style="cyan")
        table.add_column("수량", justify="right")
        table.add_column("평균가", justify="right")
        table.add_column("현재가", justify="right")
        table.add_column("평가금", justify="right")
        table.add_column("손익", justify="right")
        table.add_column("수익률", justify="right")

        for ticker, pos in summary["positions"].items():
            pnl_style = "green" if pos["pnl"] > 0 else "red" if pos["pnl"] < 0 else "white"
            table.add_row(
                f"{pos['name']}({ticker})",
                f"{pos['quantity']:,}",
                f"{pos['avg_price']:,.0f}",
                f"{pos['current_price']:,.0f}",
                f"{pos['market_value']:,.0f}",
                f"[{pnl_style}]{pos['pnl']:,.0f}[/]",
                f"[{pnl_style}]{pos['pnl_pct']}[/]",
            )
        console.print(table)
    else:
        console.print("[dim]보유 종목 없음[/]")


@app.command()
def history(ticker: str = typer.Option(None, help="종목코드 필터"), limit: int = 20):
    """매매 이력."""
    system = _get_system()
    trades = system.portfolio.get_trade_history(ticker=ticker, limit=limit)

    if not trades:
        console.print("[dim]매매 이력 없음[/]")
        return

    table = Table(title="매매 이력")
    table.add_column("시간", style="dim")
    table.add_column("종목", style="cyan")
    table.add_column("방향")
    table.add_column("수량", justify="right")
    table.add_column("가격", justify="right")
    table.add_column("손익", justify="right")

    for t in trades:
        action_style = "green" if t["action"] == "BUY" else "red"
        table.add_row(
            t["timestamp"][:16],
            f"{t['name']}({t['ticker']})",
            f"[{action_style}]{t['action']}[/]",
            f"{t['quantity']:,}",
            f"{t['price']:,.0f}",
            f"{t.get('pnl', 0) or 0:,.0f}" if t.get("pnl") else "-",
        )

    console.print(table)


# --- 자연어 전략 변경 ---

@app.command()
def chat(message: str):
    """자연어로 전략/파라미터 변경."""
    system = _get_system()
    console.print(f"[dim]분석 중: {message}[/]")

    result = system.handle_natural_language(message)

    if result.get("interpretation"):
        console.print(f"\n[bold]해석:[/] {result['interpretation']}")

    adjustments = result.get("adjustments", [])
    if adjustments:
        table = Table(title="파라미터 조정")
        table.add_column("파라미터", style="cyan")
        table.add_column("이전값")
        table.add_column("새 값", style="green")
        table.add_column("상태")
        table.add_column("사유")

        for adj in adjustments:
            status = "✅" if adj.get("applied") else "❌"
            table.add_row(
                adj["param"],
                str(adj.get("old_value", "?")),
                str(adj["value"]),
                status,
                adj.get("reason", ""),
            )
        console.print(table)


# --- ML ---

ml_app = typer.Typer(help="ML 모델 관리")
app.add_typer(ml_app, name="ml")


@ml_app.command("train")
def ml_train():
    """수동 모델 재학습."""
    system = _get_system()
    console.print("[bold]ML 모델 학습 시작...[/]")
    result = system.train_ml_models()
    console.print(json.dumps(result, ensure_ascii=False, indent=2))


@ml_app.command("predict")
def ml_predict(ticker: str):
    """종목 예측 조회."""
    system = _get_system()
    result = system.predict_ticker(ticker)
    console.print(Panel(
        json.dumps(result, ensure_ascii=False, indent=2),
        title=f"ML 예측 — {ticker}",
    ))


# --- 시뮬레이션 ---

sim_app = typer.Typer(help="시뮬레이션 관리")
app.add_typer(sim_app, name="sim")


@sim_app.command("start")
def sim_start():
    """시뮬레이션 모드 시작."""
    system = _get_system()
    system.config_manager.set_mode("simulation")
    console.print("[green]시뮬레이션 모드 활성화[/]")


# --- 리뷰 ---

review_app = typer.Typer(help="복기/리포트")
app.add_typer(review_app, name="review")


@review_app.command("today")
def review_today():
    """오늘의 복기."""
    system = _get_system()
    console.print("[bold]복기 실행 중...[/]")
    result = system.run_daily_review()
    if result:
        console.print(Panel(
            json.dumps(result, ensure_ascii=False, indent=2),
            title="오늘의 복기",
        ))


# --- 데이터 ---

data_app = typer.Typer(help="데이터 관리")
app.add_typer(data_app, name="data")


@data_app.command("collect")
def data_collect():
    """수동 데이터 수집."""
    system = _get_system()
    console.print("[bold]데이터 수집 중...[/]")
    system.collect_data()
    console.print("[green]수집 완료[/]")


if __name__ == "__main__":
    app()
