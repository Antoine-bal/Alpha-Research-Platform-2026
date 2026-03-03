import pathlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .metrics import BacktestMetrics


def export_results(
    daily_df: pd.DataFrame,
    trade_rows: List[Dict[str, Any]],
    config: BacktestConfig,
    market,  # DataStore
    strategy,  # Strategy (for entry_exit_map)
    metrics: BacktestMetrics,
    all_ptfs: Dict[int, Dict[str, Any]],
) -> None:
    """Export all backtest results to an Excel workbook."""

    if daily_df.empty:
        print("[WARN] No PnL rows to export.")
        return

    df = daily_df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    df = df.bfill()

    output_path = config.output_path

    # PORTFOLIO sheet
    pivot_perf = df.pivot(index="Date", columns="Symbol", values="Perf")
    pivot_perf["PortfolioPerf"] = pivot_perf.sum(axis=1)

    # Dispersion track (market-cap weighted — auto-fetches from Yahoo Finance if needed)
    disp_perf = None
    df_dispe_w = None
    if (config.strategy_mode in ("dispersion", "dispersion_varswap")
            and config.dispersion_component_symbols):
        disp_perf, df_dispe_w = _compute_dispersion_track(df, config)
        if disp_perf is not None:
            pivot_perf["Dispersion"] = disp_perf

    # CONFIG sheet
    config_rows = config.to_flat_dict()
    config_rows.insert(0, {"Key": "run_date", "Value": datetime.now().strftime("%Y%m%d_%H%M")})
    df_config = pd.DataFrame(config_rows, columns=["Key", "Value"])

    # EARNINGS sheet
    df_earn = market.earnings
    if df_earn.empty:
        df_earn = pd.DataFrame(columns=["Symbol", "EventDay"])
    else:
        df_earn = df_earn.sort_values(["symbol", "event_day"])
        df_earn = df_earn.rename(
            columns={"symbol": "Symbol", "event_day": "EventDay"}
        )

    # CORP_ACTIONS sheet
    df_corp = _build_corp_actions_df(config)

    # TRADES sheet
    df_trades_all = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(
            columns=[
                "Date", "Symbol", "ContractID", "Expiry", "Strike",
                "Type", "TradeQty", "Mid", "Bid", "Ask", "TradePrice",
                "TradeNotional", "Spot", "IV", "Delta", "Gamma", "Vega", "Theta",
            ]
        )
    )

    # EVENT_PNL sheet
    event_pnl_rows = []
    entry_exit_map = getattr(strategy, "entry_exit_map", {})
    for sym, mapping in entry_exit_map.items():
        for entry_date, meta in mapping.items():
            exit_date = meta["exit_date"]
            event_day = meta["event_day"]
            timing = meta.get("timing", "UNKNOWN")

            if exit_date is None:
                continue

            mask = (
                (df["Symbol"] == sym)
                & (df["Date"] >= entry_date)
                & (df["Date"] <= exit_date)
            )
            df_window = df[mask]
            if df_window.empty:
                continue

            event_pnl_rows.append({
                "Symbol": sym,
                "EntryDate": entry_date,
                "EventDay": event_day,
                "ExitDate": exit_date,
                "Timing": timing,
                "EventWindowPnL": df_window["DailyPnL"].sum(),
                "EventWindowPnL_gamma": df_window["DailyPnL"].sum() if "DailyPnL" in df_window.columns else 0.0,
            })

    df_event_pnl = (
        pd.DataFrame(event_pnl_rows)
        if event_pnl_rows
        else pd.DataFrame(
            columns=[
                "Symbol", "EntryDate", "EventDay", "ExitDate", "Timing",
                "EventWindowPnL", "EventWindowPnL_gamma",
            ]
        )
    )

    # EVENT_STATS sheet
    df_event_stats = _build_event_stats(df_event_pnl)

    # METRICS sheet (includes t-stats, CVaR, higher moments if available)
    df_metrics = metrics.to_dataframe()

    # Per-symbol metrics
    per_sym_rows = []
    if metrics.per_symbol:
        for sym, sm in metrics.per_symbol.items():
            row = {
                "Symbol": sym,
                "Total Return": sm.total_return,
                "Ann. Return": sm.annualized_return,
                "Ann. Vol": sm.annualized_vol,
                "Sharpe": sm.sharpe_ratio,
                "Sortino": sm.sortino_ratio,
                "Calmar": sm.calmar_ratio,
                "Max DD": sm.max_drawdown,
                "Max DD Duration": sm.max_drawdown_duration_days,
                "N Trades": sm.n_trades,
                "Win Rate": sm.win_rate,
                "Profit Factor": sm.profit_factor,
                "PnL Total": sm.cum_pnl_total,
                "PnL Gamma": sm.cum_pnl_gamma,
                "PnL Vega": sm.cum_pnl_vega,
                "PnL Theta": sm.cum_pnl_theta,
                "PnL Rho": sm.cum_pnl_rho,
                "PnL Residual": sm.cum_pnl_residual,
            }
            # Add enhanced metrics if available
            for attr in ["sharpe_tstat", "sharpe_pvalue", "cvar_95", "cvar_99",
                         "skewness", "kurtosis", "best_day", "worst_day", "pct_positive_days"]:
                val = getattr(sm, attr, None)
                if val is not None:
                    row[attr] = val
            per_sym_rows.append(row)
    df_per_sym_metrics = pd.DataFrame(per_sym_rows) if per_sym_rows else pd.DataFrame()

    # Write Excel
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        # 1) PORTFOLIO
        pivot_perf.reset_index().to_excel(writer, sheet_name="PORTFOLIO", index=False)

        # 2) METRICS
        df_metrics.to_excel(writer, sheet_name="METRICS", index=False)

        # 3) METRICS_BY_SYMBOL
        if not df_per_sym_metrics.empty:
            df_per_sym_metrics.to_excel(
                writer, sheet_name="METRICS_BY_SYMBOL", index=False
            )

        # 4) EVENT_PNL
        df_event_pnl.to_excel(writer, sheet_name="EVENT_PNL", index=False)

        # 5) EVENT_STATS
        df_event_stats.to_excel(writer, sheet_name="EVENT_STATS", index=False)

        # 6) CONFIG
        df_config.to_excel(writer, sheet_name="CONFIG", index=False)

        # 7) EARNINGS
        df_earn.to_excel(writer, sheet_name="EARNINGS", index=False)

        # 8) CORP_ACTIONS
        df_corp.to_excel(writer, sheet_name="CORP_ACTIONS", index=False)

        # 8b) DISPE_W (dispersion market-cap weights)
        if df_dispe_w is not None and not df_dispe_w.empty:
            df_dispe_w.to_excel(writer, sheet_name="DISPE_W", index=False)

        # 9) Per-symbol sheets
        for sym in config.symbols:
            df_sym = df[df["Symbol"] == sym]
            if not df_sym.empty:
                sheet = sym[:31]
                df_sym.to_excel(writer, sheet_name=sheet, index=False)

            if not df_trades_all.empty:
                df_tr_sym = df_trades_all[df_trades_all["Symbol"] == sym]
                if not df_tr_sym.empty:
                    sheet_tr = f"{sym}_TRADES"[:31]
                    df_tr_sym.to_excel(writer, sheet_name=sheet_tr, index=False)

    print(f"[INFO] Backtest results written to {output_path}")


def _build_event_stats(df_event_pnl: pd.DataFrame) -> pd.DataFrame:
    if df_event_pnl.empty:
        return pd.DataFrame(
            columns=[
                "Symbol", "N_events", "Mean_EventPnL", "Std_EventPnL",
                "HitRatio", "Best_EventPnL", "Worst_EventPnL",
                "BMO_N", "BMO_Mean_EventPnL", "AMC_N", "AMC_Mean_EventPnL",
            ]
        )

    base = (
        df_event_pnl.groupby("Symbol")["EventWindowPnL"]
        .agg(
            N_events="count",
            Mean_EventPnL="mean",
            Std_EventPnL="std",
            Best_EventPnL="max",
            Worst_EventPnL="min",
        )
    )

    hit = (
        df_event_pnl.assign(Positive=df_event_pnl["EventWindowPnL"] > 0)
        .groupby("Symbol")["Positive"]
        .mean()
        .rename("HitRatio")
    )

    bmo = (
        df_event_pnl[df_event_pnl["Timing"] == "BMO"]
        .groupby("Symbol")["EventWindowPnL"]
        .agg(BMO_N="count", BMO_Mean_EventPnL="mean")
    )

    amc = (
        df_event_pnl[df_event_pnl["Timing"] == "AMC"]
        .groupby("Symbol")["EventWindowPnL"]
        .agg(AMC_N="count", AMC_Mean_EventPnL="mean")
    )

    df_event_stats = (
        base.join(hit, how="left")
        .join(bmo, how="left")
        .join(amc, how="left")
        .reset_index()
    )

    for col in ["BMO_N", "AMC_N"]:
        if col in df_event_stats.columns:
            df_event_stats[col] = df_event_stats[col].fillna(0).astype(int)

    return df_event_stats


def _fetch_shares_outstanding(symbols: list) -> dict:
    """Fetch shares outstanding (in billions) from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        print("[WARN] yfinance not installed — cannot auto-fetch shares outstanding.")
        return {}

    shares = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            so = info.get("sharesOutstanding")
            if so is not None and so > 0:
                shares[sym] = so / 1e9  # convert to billions
        except Exception as e:
            print(f"[WARN] yfinance fetch failed for {sym}: {e}")
    if shares:
        print(f"[INFO] Fetched shares outstanding from Yahoo Finance for {len(shares)} symbols.")
    return shares


def _compute_dispersion_track(
    daily_df: pd.DataFrame,
    config: BacktestConfig,
):
    """Compute market-cap weighted dispersion track and weight table.

    The strategy trades each component with equal 1/N vega allocation.
    Here we rescale PnLs by (w_i * N) to get market-cap weighted aggregation,
    where w_i = market_cap_i / sum(market_caps) for the selected components.
    Weights are snapped weekly (Fridays).

    Returns (disp_perf Series, df_weights DataFrame) or (None, None).
    """
    index_sym = config.dispersion_index_symbol
    comp_syms = list(config.dispersion_component_symbols)
    N = len(comp_syms)

    # Auto-fetch from Yahoo Finance if not provided
    shares = config.dispersion_shares_outstanding
    if not shares:
        shares = _fetch_shares_outstanding(comp_syms)

    if not shares or not comp_syms or N == 0:
        return None, None

    # Pivot spot and daily PnL
    pivot_spot = daily_df.pivot_table(
        index="Date", columns="Symbol", values="Spot", aggfunc="last"
    )
    pivot_pnl = daily_df.pivot_table(
        index="Date", columns="Symbol", values="DailyPnL", aggfunc="sum"
    )
    all_dates = pivot_spot.index

    # Market caps for components: spot * shares_outstanding
    cap_df = pd.DataFrame(index=all_dates)
    for sym in comp_syms:
        if sym in pivot_spot.columns and sym in shares:
            cap_df[sym] = pivot_spot[sym] * shares[sym]

    if cap_df.empty or cap_df.dropna(how="all").empty:
        return None, None

    total_cap = cap_df.sum(axis=1)
    weights_daily = cap_df.div(total_cap, axis=0)

    # Snap to weekly: use Friday weights, forward-fill within week
    is_friday = all_dates.weekday == 4
    fridays = all_dates[is_friday]

    if len(fridays) == 0:
        return None, None

    weights_friday = weights_daily.loc[fridays]
    weekly_weights = weights_friday.reindex(all_dates, method="ffill")
    weekly_weights = weekly_weights.bfill()  # fill dates before first Friday

    # Build dispersion PnL
    # Long leg: sum(w_i * N * component_pnl_i)  — rescales from 1/N to w_i
    long_leg_pnl = pd.Series(0.0, index=all_dates)
    for sym in comp_syms:
        if sym in pivot_pnl.columns and sym in weekly_weights.columns:
            long_leg_pnl += (
                weekly_weights[sym].fillna(0) * N * pivot_pnl[sym].fillna(0)
            )

    # Short leg: index PnL (already traded with full vega, sign from strategy)
    if index_sym in pivot_pnl.columns:
        short_leg_pnl = pivot_pnl[index_sym].fillna(0)
    else:
        short_leg_pnl = 0.0

    disp_daily_pnl = long_leg_pnl + short_leg_pnl

    initial_total = config.initial_perf_per_ticker * (N + 1)
    disp_perf = initial_total + disp_daily_pnl.cumsum()

    # Build weight export table (only on Friday rebalance dates)
    weight_rows = []
    for date in fridays:
        tc = total_cap.loc[date] if date in total_cap.index else np.nan
        for sym in comp_syms:
            spot_val = (
                pivot_spot.loc[date, sym]
                if sym in pivot_spot.columns and date in pivot_spot.index
                else np.nan
            )
            cap_val = (
                cap_df.loc[date, sym]
                if sym in cap_df.columns and date in cap_df.index
                else np.nan
            )
            w_val = (
                weekly_weights.loc[date, sym]
                if sym in weekly_weights.columns and date in weekly_weights.index
                else np.nan
            )
            weight_rows.append({
                "Date": date,
                "Symbol": sym,
                "Spot": spot_val,
                "SharesOut_B": shares.get(sym, np.nan),
                "MarketCap_B": cap_val,
                "TotalCap_B": tc,
                "Weight": w_val,
            })

    df_weights = pd.DataFrame(weight_rows)
    return disp_perf, df_weights


def _build_corp_actions_df(config: BacktestConfig) -> pd.DataFrame:
    corp_dir = pathlib.Path(config.corp_dir)
    corp_rows = []

    for sym in config.symbols:
        path = corp_dir / f"{sym}_daily_adjusted.parquet"
        if not path.exists():
            continue

        df_c = pd.read_parquet(path)
        if "date" not in df_c.columns:
            continue

        df_c["date"] = pd.to_datetime(df_c["date"]).dt.normalize()
        df_c = df_c[
            (df_c["date"] >= config.start_date) & (df_c["date"] <= config.end_date)
        ]
        if df_c.empty:
            continue

        lower_cols = {c.lower(): c for c in df_c.columns}
        div_col = None
        split_col = None
        for lc, orig in lower_cols.items():
            if "dividend" in lc:
                div_col = orig
            if "split" in lc:
                split_col = orig

        if div_col is None and split_col is None:
            continue

        mask = False
        if div_col is not None:
            mask = df_c[div_col].fillna(0).astype(float) != 0
        if split_col is not None:
            split_mask = df_c[split_col].fillna(1).astype(float) != 1
            mask = mask | split_mask if isinstance(mask, pd.Series) else split_mask

        df_c = df_c[mask]
        if df_c.empty:
            continue

        cols = ["date"]
        if div_col is not None:
            cols.append(div_col)
        if split_col is not None:
            cols.append(split_col)

        df_c = df_c[cols]
        df_c["Symbol"] = sym

        rename_map = {"date": "Date"}
        if div_col is not None:
            rename_map[div_col] = "Dividend"
        if split_col is not None:
            rename_map[split_col] = "SplitFactor"

        df_c = df_c.rename(columns=rename_map)
        corp_rows.append(df_c)

    if not corp_rows:
        return pd.DataFrame(columns=["Symbol", "Date", "Dividend", "SplitFactor"])

    df_corp = pd.concat(corp_rows, ignore_index=True)
    for col in ["Dividend", "SplitFactor"]:
        if col not in df_corp.columns:
            df_corp[col] = np.nan

    df_corp = df_corp[["Symbol", "Date", "Dividend", "SplitFactor"]]
    df_corp = df_corp.sort_values(["Symbol", "Date"])
    return df_corp
