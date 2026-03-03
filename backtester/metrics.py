from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestMetrics:
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Risk
    annualized_vol: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Ratios
    sharpe_ratio: float = 0.0
    sharpe_tstat: float = 0.0
    sharpe_pvalue: float = 1.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Higher moments
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Daily extremes
    best_day: float = 0.0
    worst_day: float = 0.0
    pct_positive_days: float = 0.0

    # Trade-level
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    n_trades: int = 0

    # Attribution
    cum_pnl_total: float = 0.0
    cum_pnl_gamma: float = 0.0
    cum_pnl_vega: float = 0.0
    cum_pnl_theta: float = 0.0
    cum_pnl_rho: float = 0.0
    cum_pnl_residual: float = 0.0
    cum_pnl_delta_hedge: float = 0.0
    cum_pnl_tc: float = 0.0

    # Per-symbol
    per_symbol: Optional[Dict[str, "BacktestMetrics"]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a DataFrame suitable for Excel export.

        All values are raw floats (no string formatting) so that
        downstream aggregation and analysis can treat them numerically.
        """
        rows = [
            ("Total Return", self.total_return),
            ("Annualized Return", self.annualized_return),
            ("Annualized Volatility", self.annualized_vol),
            ("Sharpe Ratio", self.sharpe_ratio),
            ("Sharpe t-stat (Lo 2002)", self.sharpe_tstat),
            ("Sharpe p-value", self.sharpe_pvalue),
            ("Sortino Ratio", self.sortino_ratio),
            ("Calmar Ratio", self.calmar_ratio),
            ("CVaR 95%", self.cvar_95),
            ("CVaR 99%", self.cvar_99),
            ("Skewness", self.skewness),
            ("Excess Kurtosis", self.kurtosis),
            ("Best Day", self.best_day),
            ("Worst Day", self.worst_day),
            ("% Positive Days", self.pct_positive_days),
            ("Max Drawdown", self.max_drawdown),
            ("Max DD Duration (days)", self.max_drawdown_duration_days),
            ("N Trades", self.n_trades),
            ("Win Rate", self.win_rate),
            ("Profit Factor", self.profit_factor),
            ("Avg Win", self.avg_win),
            ("Avg Loss", self.avg_loss),
            ("Cum PnL Total", self.cum_pnl_total),
            ("Cum PnL Gamma", self.cum_pnl_gamma),
            ("Cum PnL Vega", self.cum_pnl_vega),
            ("Cum PnL Theta", self.cum_pnl_theta),
            ("Cum PnL Rho", self.cum_pnl_rho),
            ("Cum PnL Residual", self.cum_pnl_residual),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])


def _compute_enhanced_risk_metrics(daily_returns: pd.Series) -> dict:
    """Compute t-stat, CVaR, higher moments from daily returns series.

    Uses Lo (2002) adjustment for non-IID returns in Sharpe t-stat.
    """
    n = len(daily_returns)
    if n < 10:
        return {}

    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std())
    sr_daily = mu / sigma if sigma > 1e-10 else 0.0

    # Higher moments
    s3 = float(daily_returns.skew())  # skewness
    s4 = float(daily_returns.kurtosis())  # excess kurtosis

    # Lo (2002) Sharpe ratio t-statistic with non-normality adjustment
    denom_sq = (1.0 + 0.5 * s3 * sr_daily - (s4 / 4.0) * sr_daily ** 2) / n
    if denom_sq > 0:
        se = np.sqrt(denom_sq)
        tstat = sr_daily / se if se > 1e-10 else 0.0
    else:
        tstat = sr_daily * np.sqrt(n)
    pvalue = float(2.0 * stats.t.sf(abs(tstat), df=max(n - 1, 1)))

    # CVaR (Expected Shortfall) — annualized
    sorted_rets = daily_returns.sort_values()
    cutoff_95 = max(int(n * 0.05), 1)
    cutoff_99 = max(int(n * 0.01), 1)
    cvar_95 = float(sorted_rets.iloc[:cutoff_95].mean() * np.sqrt(252))
    cvar_99 = float(sorted_rets.iloc[:cutoff_99].mean() * np.sqrt(252))

    return {
        "sharpe_tstat": tstat,
        "sharpe_pvalue": pvalue,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "skewness": s3,
        "kurtosis": s4,
        "best_day": float(daily_returns.max()),
        "worst_day": float(daily_returns.min()),
        "pct_positive_days": float((daily_returns > 0).mean()),
    }


def compute_metrics(
    daily_df: pd.DataFrame,
    event_df: Optional[pd.DataFrame] = None,
) -> BacktestMetrics:
    """
    Compute standard performance metrics from backtester output.

    Parameters
    ----------
    daily_df : DataFrame with columns Date, Symbol, Perf, DailyPnL, PnL_*
    event_df : Optional DataFrame with EventWindowPnL column (for trade-level metrics)
    """
    if daily_df.empty:
        return BacktestMetrics()

    # Portfolio perf curve (sum across symbols)
    pivot_perf = daily_df.pivot_table(
        index="Date", columns="Symbol", values="Perf", aggfunc="first"
    )
    portfolio_perf = pivot_perf.sum(axis=1).sort_index()

    if len(portfolio_perf) < 2:
        return BacktestMetrics()

    daily_returns = portfolio_perf.pct_change().dropna()
    n_days = len(daily_returns)

    # Total return
    total_return = (portfolio_perf.iloc[-1] / portfolio_perf.iloc[0]) - 1

    # Annualized return (compound)
    n_years = n_days / 252.0
    if n_years > 0:
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0

    # Annualized vol
    annualized_vol = float(daily_returns.std() * np.sqrt(252))

    # Sharpe (excess return / vol, assuming rf=0 for simplicity)
    sharpe = annualized_return / annualized_vol if annualized_vol > 1e-10 else 0.0

    # Sortino (downside deviation)
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1:
        downside_std = float(downside.std() * np.sqrt(252))
    else:
        downside_std = annualized_vol
    sortino = annualized_return / downside_std if downside_std > 1e-10 else 0.0

    # Max drawdown
    cummax = portfolio_perf.cummax()
    drawdown = (portfolio_perf - cummax) / cummax
    max_dd = float(drawdown.min())

    # Max drawdown duration
    underwater = drawdown < -1e-10
    if underwater.any():
        # Count consecutive underwater days
        groups = (~underwater).cumsum()
        underwater_groups = underwater.groupby(groups).sum()
        max_dd_duration = int(underwater_groups.max())
    else:
        max_dd_duration = 0

    # Calmar
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Trade-level metrics
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    n_trades = 0

    if event_df is not None and not event_df.empty and "EventWindowPnL" in event_df.columns:
        pnls = event_df["EventWindowPnL"].dropna()
        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            n_trades = len(pnls)
            win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
            total_wins = wins.sum() if len(wins) > 0 else 0.0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = (
                total_wins / total_losses if total_losses > 1e-10 else float("inf")
            )
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    # Enhanced risk metrics: t-stat, CVaR, higher moments
    enhanced = _compute_enhanced_risk_metrics(daily_returns)

    # PnL attribution
    cum_pnl_total = float(daily_df["DailyPnL"].sum())
    _last = daily_df.iloc[-1] if len(daily_df) > 0 else {}
    cum_pnl_gamma = float(_last.get("CumPnL_gamma", 0.0))
    cum_pnl_vega = float(_last.get("CumPnL_vega", 0.0))
    cum_pnl_theta = float(_last.get("CumPnL_theta", 0.0))
    cum_pnl_rho = float(_last.get("CumPnL_rho", 0.0))
    cum_pnl_residual = float(_last.get("CumPnL_residual", 0.0))
    cum_pnl_delta_hedge = float(_last.get("CumPnL_deltaHedge", 0.0))
    cum_pnl_tc = float(_last.get("CumPnL_TC", 0.0))

    # Per-symbol metrics
    per_symbol: Dict[str, BacktestMetrics] = {}
    for sym in daily_df["Symbol"].unique():
        sym_df = daily_df[daily_df["Symbol"] == sym]
        sym_event_df = None
        if event_df is not None and not event_df.empty and "Symbol" in event_df.columns:
            sym_event_df = event_df[event_df["Symbol"] == sym]
        per_symbol[sym] = _compute_single_symbol_metrics(sym_df, sym_event_df)

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        sharpe_ratio=sharpe,
        sharpe_tstat=enhanced.get("sharpe_tstat", 0.0),
        sharpe_pvalue=enhanced.get("sharpe_pvalue", 1.0),
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        cvar_95=enhanced.get("cvar_95", 0.0),
        cvar_99=enhanced.get("cvar_99", 0.0),
        skewness=enhanced.get("skewness", 0.0),
        kurtosis=enhanced.get("kurtosis", 0.0),
        best_day=enhanced.get("best_day", 0.0),
        worst_day=enhanced.get("worst_day", 0.0),
        pct_positive_days=enhanced.get("pct_positive_days", 0.0),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        n_trades=n_trades,
        cum_pnl_total=cum_pnl_total,
        cum_pnl_gamma=cum_pnl_gamma,
        cum_pnl_vega=cum_pnl_vega,
        cum_pnl_theta=cum_pnl_theta,
        cum_pnl_rho=cum_pnl_rho,
        cum_pnl_residual=cum_pnl_residual,
        cum_pnl_delta_hedge=cum_pnl_delta_hedge,
        cum_pnl_tc=cum_pnl_tc,
        per_symbol=per_symbol,
    )


def _compute_single_symbol_metrics(
    sym_df: pd.DataFrame, event_df: Optional[pd.DataFrame]
) -> BacktestMetrics:
    """Compute metrics for a single symbol."""
    perf = sym_df.set_index("Date")["Perf"].sort_index()

    if len(perf) < 2:
        return BacktestMetrics()

    daily_returns = perf.pct_change().dropna()
    n_days = len(daily_returns)

    total_return = (perf.iloc[-1] / perf.iloc[0]) - 1
    n_years = n_days / 252.0
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    annualized_vol = float(daily_returns.std() * np.sqrt(252))
    sharpe = annualized_return / annualized_vol if annualized_vol > 1e-10 else 0.0

    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else annualized_vol
    sortino = annualized_return / downside_std if downside_std > 1e-10 else 0.0

    cummax = perf.cummax()
    drawdown = (perf - cummax) / cummax
    max_dd = float(drawdown.min())
    calmar = annualized_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    underwater = drawdown < -1e-10
    if underwater.any():
        groups = (~underwater).cumsum()
        max_dd_duration = int(underwater.groupby(groups).sum().max())
    else:
        max_dd_duration = 0

    # Enhanced risk metrics
    enhanced = _compute_enhanced_risk_metrics(daily_returns)

    # Trade-level
    win_rate = profit_factor = avg_win = avg_loss = 0.0
    n_trades = 0
    if event_df is not None and not event_df.empty and "EventWindowPnL" in event_df.columns:
        pnls = event_df["EventWindowPnL"].dropna()
        if len(pnls) > 0:
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            n_trades = len(pnls)
            win_rate = len(wins) / n_trades
            total_wins = wins.sum() if len(wins) > 0 else 0.0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = total_wins / total_losses if total_losses > 1e-10 else float("inf")
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        sharpe_ratio=sharpe,
        sharpe_tstat=enhanced.get("sharpe_tstat", 0.0),
        sharpe_pvalue=enhanced.get("sharpe_pvalue", 1.0),
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        cvar_95=enhanced.get("cvar_95", 0.0),
        cvar_99=enhanced.get("cvar_99", 0.0),
        skewness=enhanced.get("skewness", 0.0),
        kurtosis=enhanced.get("kurtosis", 0.0),
        best_day=enhanced.get("best_day", 0.0),
        worst_day=enhanced.get("worst_day", 0.0),
        pct_positive_days=enhanced.get("pct_positive_days", 0.0),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        n_trades=n_trades,
        cum_pnl_total=float(sym_df["DailyPnL"].sum()),
        cum_pnl_gamma=float(sym_df.iloc[-1].get("CumPnL_gamma", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_vega=float(sym_df.iloc[-1].get("CumPnL_vega", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_theta=float(sym_df.iloc[-1].get("CumPnL_theta", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_rho=float(sym_df.iloc[-1].get("CumPnL_rho", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_residual=float(sym_df.iloc[-1].get("CumPnL_residual", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_delta_hedge=float(sym_df.iloc[-1].get("CumPnL_deltaHedge", 0.0)) if len(sym_df) > 0 else 0.0,
        cum_pnl_tc=float(sym_df.iloc[-1].get("CumPnL_TC", 0.0)) if len(sym_df) > 0 else 0.0,
    )
