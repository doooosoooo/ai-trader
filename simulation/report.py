"""백테스트 리포트 — 콘솔(rich) + 텔레그램."""

from rich.console import Console
from rich.table import Table

from simulation.backtest import BacktestResult


def print_console_report(result: BacktestResult) -> None:
    """터미널에 백테스트 결과 출력."""
    console = Console()
    m = result.metrics

    if "error" in m:
        console.print(f"[red]백테스트 실패: {m['error']}[/red]")
        return

    # 헤더
    console.print()
    console.rule(f"[bold]백테스트 결과: {result.strategy_name}[/bold]")
    console.print(f"기간: {result.period}")
    console.print(f"종목: {', '.join(result.tickers)}")
    console.print()

    # 수익 지표 테이블
    perf = Table(title="수익 지표", show_header=True, header_style="bold cyan")
    perf.add_column("지표", style="bold")
    perf.add_column("값", justify="right")

    total_ret = m.get("total_return", 0)
    ret_color = "green" if total_ret >= 0 else "red"
    perf.add_row("초기자본", f"{m.get('initial_capital', 0):,.0f}원")
    perf.add_row("최종자산", f"{m.get('final_value', 0):,.0f}원")
    perf.add_row("총수익률", f"[{ret_color}]{total_ret:+.2%}[/{ret_color}]")
    perf.add_row("연환산수익률", f"{m.get('annualized_return', 0):+.2%}")
    perf.add_row("MDD", f"[red]-{m.get('max_drawdown', 0):.2%}[/red]")
    perf.add_row("샤프비율", f"{m.get('sharpe_ratio', 0):.2f}")
    perf.add_row("소르티노비율", f"{m.get('sortino_ratio', 0):.2f}")
    perf.add_row("칼마비율", f"{m.get('calmar_ratio', 0):.2f}")

    if "benchmark_return" in m:
        bm_ret = m["benchmark_return"]
        perf.add_row("벤치마크(B&H)", f"{bm_ret:+.2%}")
        excess = m.get("excess_return", 0)
        ex_color = "green" if excess >= 0 else "red"
        perf.add_row("초과수익", f"[{ex_color}]{excess:+.2%}[/{ex_color}]")

    console.print(perf)
    console.print()

    # 매매 통계 테이블
    trades = Table(title="매매 통계", show_header=True, header_style="bold cyan")
    trades.add_column("지표", style="bold")
    trades.add_column("값", justify="right")

    trades.add_row("총 매매 수", f"{m.get('total_trades', 0)}")
    trades.add_row("매도 거래", f"{m.get('total_sells', 0)}")
    trades.add_row("승리", f"{m.get('winning_trades', 0)}")
    trades.add_row("패배", f"{m.get('losing_trades', 0)}")
    trades.add_row("승률", f"{m.get('win_rate', 0):.1%}")
    trades.add_row("수익팩터", f"{m.get('profit_factor', 0):.2f}")
    trades.add_row("평균 보유일", f"{m.get('avg_hold_days', 0):.1f}일")
    trades.add_row("평균 수익(승)", f"{m.get('avg_win_pct', 0):+.2%}")
    trades.add_row("평균 손실(패)", f"{m.get('avg_loss_pct', 0):+.2%}")
    trades.add_row("최대 연승", f"{m.get('max_consecutive_wins', 0)}")
    trades.add_row("최대 연패", f"{m.get('max_consecutive_losses', 0)}")

    console.print(trades)
    console.print()

    # 최근 매매 내역 (최대 15건)
    if result.trades:
        trade_list = Table(title="매매 내역 (최근 15건)", show_header=True, header_style="bold cyan")
        trade_list.add_column("날짜")
        trade_list.add_column("종목")
        trade_list.add_column("액션")
        trade_list.add_column("수량", justify="right")
        trade_list.add_column("가격", justify="right")
        trade_list.add_column("손익", justify="right")
        trade_list.add_column("사유")

        for t in result.trades[-15:]:
            action_style = "green" if t.action == "BUY" else "red"
            pnl_str = ""
            if t.action == "SELL":
                pnl_color = "green" if t.pnl >= 0 else "red"
                pnl_str = f"[{pnl_color}]{t.pnl:+,.0f} ({t.pnl_pct:+.2%})[/{pnl_color}]"
            trade_list.add_row(
                t.date, t.ticker,
                f"[{action_style}]{t.action}[/{action_style}]",
                f"{t.quantity:,}", f"{t.price:,.0f}",
                pnl_str, t.reason or "",
            )

        console.print(trade_list)

    # 전략 파라미터
    console.print()
    console.print("[dim]전략 파라미터:[/dim]")
    for k, v in result.params.items():
        if k not in ("entry_conditions", "exit_conditions"):
            console.print(f"  [dim]{k}: {v}[/dim]")

    console.print()


def print_comparison_report(results: list[BacktestResult]) -> None:
    """여러 전략 비교 테이블 출력."""
    console = Console()
    console.print()
    console.rule("[bold]전략 비교[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("전략", style="bold")
    table.add_column("수익률", justify="right")
    table.add_column("연환산", justify="right")
    table.add_column("MDD", justify="right")
    table.add_column("샤프", justify="right")
    table.add_column("승률", justify="right")
    table.add_column("수익팩터", justify="right")
    table.add_column("매매수", justify="right")

    for r in results:
        m = r.metrics
        if "error" in m:
            table.add_row(r.strategy_name, "[red]ERROR[/red]", "", "", "", "", "", "")
            continue

        ret = m.get("total_return", 0)
        ret_color = "green" if ret >= 0 else "red"
        table.add_row(
            r.strategy_name,
            f"[{ret_color}]{ret:+.2%}[/{ret_color}]",
            f"{m.get('annualized_return', 0):+.2%}",
            f"-{m.get('max_drawdown', 0):.2%}",
            f"{m.get('sharpe_ratio', 0):.2f}",
            f"{m.get('win_rate', 0):.1%}",
            f"{m.get('profit_factor', 0):.2f}",
            f"{m.get('total_trades', 0)}",
        )

    console.print(table)
    console.print()


def format_telegram_report(result: BacktestResult) -> str:
    """텔레그램 메시지 포맷."""
    m = result.metrics
    if "error" in m:
        return f"백테스트 실패: {m['error']}"

    ret = m.get("total_return", 0)
    ret_emoji = "📈" if ret >= 0 else "📉"

    msg = (
        f"📊 <b>백테스트 결과: {result.strategy_name}</b>\n"
        f"━━━━━━━━━━━━━━\n"
        f"기간: {result.period}\n"
        f"종목: {', '.join(result.tickers)}\n\n"
        f"{ret_emoji} 총수익률: {ret:+.2%}\n"
        f"📅 연환산: {m.get('annualized_return', 0):+.2%}\n"
        f"⬇️ MDD: -{m.get('max_drawdown', 0):.2%}\n"
        f"📐 샤프비율: {m.get('sharpe_ratio', 0):.2f}\n"
        f"🎯 승률: {m.get('win_rate', 0):.1%} ({m.get('winning_trades', 0)}/{m.get('total_sells', 0)})\n"
        f"⚖️ 수익팩터: {m.get('profit_factor', 0):.2f}\n"
        f"📆 평균보유: {m.get('avg_hold_days', 0):.1f}일\n"
    )

    if "benchmark_return" in m:
        excess = m.get("excess_return", 0)
        ex_emoji = "✅" if excess >= 0 else "❌"
        msg += (
            f"\n벤치마크(B&H): {m['benchmark_return']:+.2%}\n"
            f"{ex_emoji} 초과수익: {excess:+.2%}\n"
        )

    return msg


def format_optimization_report(results: list[dict]) -> str:
    """파라미터 최적화 결과 텔레그램 포맷."""
    if not results:
        return "최적화 결과 없음"

    msg = "🔧 <b>파라미터 최적화 결과</b>\n━━━━━━━━━━━━━━\n\n"

    for i, r in enumerate(results[:5], 1):
        params = r.get("params", {})
        m = r.get("metrics", {})
        msg += (
            f"<b>#{i}</b> 샤프={m.get('sharpe_ratio', 0):.2f} "
            f"수익={m.get('total_return', 0):+.2%} "
            f"MDD=-{m.get('max_drawdown', 0):.2%}\n"
        )
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        msg += f"  └ {param_str}\n\n"

    return msg
