"""
Entry point for running backtests with the refactored backtester package.

Usage:
    python run_backtest.py

Edit the config below to change strategy parameters.
"""

import pandas as pd

from backtester import BacktestConfig, BacktestEngine


# ============================================================
# Configuration
# ============================================================

SYMBOLS = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]

config = BacktestConfig(
    symbols=SYMBOLS,
    start_date=pd.Timestamp("2020-01-01"),
    end_date=pd.Timestamp("2025-11-16"),
    output_path=r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs\earnings_TermSlope_filtered.xlsx",

    # Portfolio
    initial_perf_per_ticker=100.0,
    reinvest=True,
    base_vega_target=1.0 / 20,

    # Strategy
    strategy_mode="earnings",

    # Earnings sub-config
    earnings_structure="straddle",
    strangle_mny_offset=0.035,

    # Rolling sub-config
    rolling_leg_type="P",
    rolling_select_by="delta",
    rolling_target_delta=-0.20,
    rolling_maturity_index=3,
    rolling_expiry_selection={"mode": "min_dte", "min_dte": 20},
    rolling_min_dte=1,
    rolling_max_dte=365,
    rolling_holding_days=20,
    rolling_direction=1.0,

    # Signals — TermSlope threshold filter
    use_signal=True,
    signal_mode="short",
    signal_filter_col="TermSlope_pre_z",
    signal_filter_min=0.0,  # trade only above-average TermSlope (top ~50%)

    # PnL Explain — independent-bump BS repricing attribution
    pnl_explain=True,
    pnl_explain_n_steps=10,

    # Execution
    execution_mode="mid_spread",
    cost_model={
        "option_spread_bps": 50,
        "stock_spread_bps": 1,
        "commission_per_contract": 0.0,
    },
    delta_hedge=True,

    # Data quality
    min_moneyness=0.5,
    max_moneyness=1.5,
    min_dte_for_entry=5,
    max_dte_for_entry=30,
    max_bid_ask_spread_pct=0.5,

    # Exit
    exit_fallback_mode="intrinsic",
    trade_log_mode="all",
)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    engine = BacktestEngine(config)
    metrics = engine.run()

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:       {metrics.total_return:.2%}")
    print(f"  Annualized Return:  {metrics.annualized_return:.2%}")
    print(f"  Annualized Vol:     {metrics.annualized_vol:.2%}")
    print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:.3f}")
    print(f"  Sharpe t-stat:      {metrics.sharpe_tstat:.3f}  (p={metrics.sharpe_pvalue:.4f})")
    print(f"  Sortino Ratio:      {metrics.sortino_ratio:.3f}")
    print(f"  Calmar Ratio:       {metrics.calmar_ratio:.3f}")
    print(f"  CVaR 95%:           {metrics.cvar_95:.4f}")
    print(f"  CVaR 99%:           {metrics.cvar_99:.4f}")
    print(f"  Skewness:           {metrics.skewness:.3f}")
    print(f"  Excess Kurtosis:    {metrics.kurtosis:.3f}")
    print(f"  Max Drawdown:       {metrics.max_drawdown:.2%}")
    print(f"  Max DD Duration:    {metrics.max_drawdown_duration_days} days")
    print(f"  Best Day:           {metrics.best_day:.4f}")
    print(f"  Worst Day:          {metrics.worst_day:.4f}")
    print(f"  % Positive Days:    {metrics.pct_positive_days:.1%}")
    print(f"  Win Rate:           {metrics.win_rate:.1%}")
    print(f"  Profit Factor:      {metrics.profit_factor:.3f}")
    print(f"  N Trades:           {metrics.n_trades}")
    print()
    print("  PnL Attribution:")
    print(f"    Total:       {metrics.cum_pnl_total:.6f}")
    print(f"    Gamma:              {metrics.cum_pnl_gamma:.6f}")
    print(f"    Vega:               {metrics.cum_pnl_vega:.6f}")
    print(f"    Theta:              {metrics.cum_pnl_theta:.6f}")
    print(f"    Rho:                {metrics.cum_pnl_rho:.6f}")
    print(f"    Vanna:              {metrics.cum_pnl_vanna:.6f}")
    print(f"    Charm:              {metrics.cum_pnl_charm:.6f}")
    print(f"    Veta:               {metrics.cum_pnl_veta:.6f}")
    print(f"    Residual:           {metrics.cum_pnl_residual:.6f}")
    print("=" * 60)
