"""
Flagship Portfolio Quality Metrics.

Loads all strategy outputs, computes:
1. Individual strategy metrics (Sharpe, Sortino, MaxDD, etc.)
2. Composite portfolio (carry 60%, hedge 40%)
3. Diversification quality (Effective N, diversification ratio, correlations)
4. Stress behavior (crisis beta, conditional correlation)
5. Regime robustness (bull/bear/range Sharpe, regime coverage)
6. Marginal value (marginal Sharpe contribution per strategy)

Usage:
    python flagship_metrics.py             # defaults to V4
    python flagship_metrics.py --version 3 # for V3
    python flagship_metrics.py --version 2 # for V2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Defaults — overridden by --version CLI arg
OUTPUT_DIR = Path("outputs/flagship_suite_v3")
FILE_PREFIX = "FS"

# ================================================================
# Strategy classification for V2
# ================================================================
CARRY = ["DVRP", "SVRP", "PRSK", "PSPD", "TCAL", "TSRV", "IRON", "NKPT", "VMRV", "BFLY"]
HEDGE = ["VLNG", "LVSP", "PRB", "RCAL", "V9HG", "DHOV"]

ALLOC = {
    "carry":  (CARRY, 0.60),
    "hedge":  (HEDGE, 0.40),
}


# ================================================================
# Data loading
# ================================================================

def load_perf(code: str, symbol: str = "QQQ") -> pd.Series:
    """Load daily perf from Excel PORTFOLIO sheet."""
    path = OUTPUT_DIR / f"{FILE_PREFIX}_{code}_{symbol}.xlsx"
    if not path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_excel(path, sheet_name="PORTFOLIO", engine="openpyxl")
        if "PortfolioPerf" in df.columns:
            df.index = pd.to_datetime(df.iloc[:, 0])
            return df["PortfolioPerf"]
        date_col = df.columns[0]
        perf_col = [c for c in df.columns if c != date_col][0]
        s = df.set_index(date_col)[perf_col]
        s.index = pd.to_datetime(s.index)
        return s
    except Exception as e:
        print(f"  Warning: Could not load {code}: {e}")
        return pd.Series(dtype=float)


def load_spot(symbol: str = "QQQ") -> pd.Series:
    """Load spot for given symbol."""
    # Try Alpha Vantage format first
    spot_path = Path(f"alpha_corp_actions/{symbol}_daily_adjusted.parquet")
    if not spot_path.exists():
        # Try OptionMetrics format (for NDX, SPX)
        spot_path = Path(f"om_data/spot_data/{symbol}.parquet")
    if not spot_path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(spot_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    for col in ["adjusted_close", "close", "spot"]:
        if col in df.columns:
            return df[col]
    return df.iloc[:, 0]


# ================================================================
# Metrics functions
# ================================================================

def compute_metrics(perf: pd.Series) -> dict:
    """Standard backtest metrics from perf series (starting ~100)."""
    daily_ret = perf.pct_change().dropna()
    if len(daily_ret) < 20:
        return {}
    total_ret = perf.iloc[-1] / perf.iloc[0] - 1
    years = len(daily_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = perf.cummax()
    dd = (perf - cummax) / cummax
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0
    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    win_rate = (daily_ret > 0).mean()
    # Skewness and kurtosis of returns
    skew = daily_ret.skew()
    kurt = daily_ret.kurtosis()
    return {
        "Total Return": total_ret,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_dd,
        "Win Rate": win_rate,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Days": len(daily_ret),
    }


def compute_effective_n(ret_df: pd.DataFrame) -> float:
    """Effective N from PCA eigenvalues: sum(lambda)^2 / sum(lambda^2)."""
    cov = ret_df.cov()
    eigenvalues = np.linalg.eigvalsh(cov.values)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()


def compute_diversification_ratio(ret_df: pd.DataFrame, weights: pd.Series) -> float:
    """Diversification ratio: sum(w_i * sigma_i) / sigma_portfolio."""
    vols = ret_df.std() * np.sqrt(252)
    w = weights.reindex(ret_df.columns).fillna(0)
    weighted_vol_sum = (w * vols).sum()
    ptf_ret = (ret_df * w).sum(axis=1)
    ptf_vol = ptf_ret.std() * np.sqrt(252)
    if ptf_vol > 0:
        return weighted_vol_sum / ptf_vol
    return 1.0


def compute_conditional_correlation(ret_df: pd.DataFrame, threshold_pctl=0.10):
    """Average pairwise correlation on stress days vs calm days."""
    ptf_ret = ret_df.mean(axis=1)
    threshold = ptf_ret.quantile(threshold_pctl)
    stress_days = ret_df[ptf_ret <= threshold]
    calm_days = ret_df[ptf_ret > threshold]
    stress_corr = stress_days.corr().values
    calm_corr = calm_days.corr().values
    np.fill_diagonal(stress_corr, np.nan)
    np.fill_diagonal(calm_corr, np.nan)
    return np.nanmean(stress_corr), np.nanmean(calm_corr)


def compute_crisis_beta(ret_df: pd.DataFrame, spot: pd.Series, threshold=-0.01):
    """Portfolio beta vs QQQ on days QQQ drops > threshold."""
    spot_ret = spot.pct_change().dropna()
    common = ret_df.index.intersection(spot_ret.index)
    ptf_ret = ret_df.loc[common].mean(axis=1)
    spot_r = spot_ret.loc[common]
    stress = spot_r < threshold
    if stress.sum() < 10:
        return np.nan
    from numpy.linalg import lstsq
    x = spot_r[stress].values.reshape(-1, 1)
    y = ptf_ret[stress].values
    coef, _, _, _ = lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)
    return coef[0]


def compute_regime_sharpes(perf: pd.Series, spot: pd.Series):
    """Compute Sharpe in bull, bear, and range regimes."""
    daily_ret = perf.pct_change().dropna()
    sma200 = spot.rolling(200).mean()
    ret_20d = spot.pct_change(20)
    common = daily_ret.index.intersection(sma200.dropna().index).intersection(ret_20d.dropna().index)
    if len(common) < 100:
        return np.nan, np.nan, np.nan
    daily_ret = daily_ret.loc[common]
    sma = sma200.loc[common]
    sp = spot.loc[common]
    r20 = ret_20d.loc[common]

    bull = (sp > sma) & (r20 > 0.02)
    bear = (sp < sma) & (r20 < -0.02)
    ranging = ~bull & ~bear

    def _sharpe(r):
        if len(r) < 20:
            return np.nan
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0

    return _sharpe(daily_ret[bull]), _sharpe(daily_ret[bear]), _sharpe(daily_ret[ranging])


def compute_marginal_sharpe(ret_df: pd.DataFrame, weights: pd.Series):
    """For each strategy, compute Sharpe(full) - Sharpe(full minus this strategy)."""
    w = weights.reindex(ret_df.columns).fillna(0)
    full_ret = (ret_df * w).sum(axis=1)
    full_sharpe = full_ret.mean() / full_ret.std() * np.sqrt(252) if full_ret.std() > 0 else 0

    marginal = {}
    for col in ret_df.columns:
        w_ex = w.copy()
        w_ex[col] = 0
        # Rescale remaining weights to sum to 1
        if w_ex.sum() > 0:
            w_ex = w_ex / w_ex.sum()
        ex_ret = (ret_df * w_ex).sum(axis=1)
        ex_sharpe = ex_ret.mean() / ex_ret.std() * np.sqrt(252) if ex_ret.std() > 0 else 0
        marginal[col] = full_sharpe - ex_sharpe
    return marginal


# ================================================================
# Yearly returns
# ================================================================

def compute_yearly_returns(ret_df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual returns for each strategy and composite."""
    yearly = {}
    for col in ret_df.columns:
        for year in sorted(ret_df.index.year.unique()):
            mask = ret_df.index.year == year
            yr_ret = (1 + ret_df.loc[mask, col]).prod() - 1
            yearly.setdefault(col, {})[year] = yr_ret
    return pd.DataFrame(yearly)


# ================================================================
# Main
# ================================================================

def main(symbol: str = "QQQ", opti: bool = False, start_date: str = None):
    print("=" * 90)
    print(f"FLAGSHIP — PORTFOLIO QUALITY METRICS ({symbol})")
    print("=" * 90)

    # Load all strategies
    all_perfs = {}
    for code in CARRY + HEDGE:
        s = load_perf(code, symbol)
        if not s.empty:
            all_perfs[code] = s
            cat = "carry" if code in CARRY else "hedge"
            print(f"  {code:8s} ({cat:6s}): {len(s)} days, last={s.iloc[-1]:.2f}")
        else:
            print(f"  {code:8s}: NO DATA")

    if not all_perfs:
        print("No data loaded!")
        return

    # Truncate to start_date if specified
    if start_date:
        cutoff = pd.Timestamp(start_date)
        trimmed = {}
        for code, s in all_perfs.items():
            s2 = s[s.index >= cutoff]
            if not s2.empty:
                trimmed[code] = s2 / s2.iloc[0] * 100  # rebase to 100
        all_perfs = trimmed
        print(f"\n  [FILTER] Truncated to >= {start_date} ({len(next(iter(all_perfs.values())))} days)")

    # Align all series
    perf_df = pd.DataFrame(all_perfs)
    perf_df = perf_df.dropna(how="all").ffill()
    ret_df = perf_df.pct_change().fillna(0)

    # ================================================================
    # 1. Individual strategy metrics
    # ================================================================
    print(f"\n\n{'='*90}")
    print("INDIVIDUAL STRATEGY METRICS")
    print(f"{'='*90}")
    print(f"{'Code':8s} {'Cat':6s} {'Return':>8s} {'AnnRet':>8s} {'AnnVol':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Skew':>7s}")
    print("-" * 85)

    metrics_all = {}
    for code, perf in all_perfs.items():
        m = compute_metrics(perf)
        metrics_all[code] = m
        cat = "carry" if code in CARRY else "hedge"
        if m:
            print(f"{code:8s} {cat:6s} {m['Total Return']:>+7.1%} {m['Ann. Return']:>+7.1%} "
                  f"{m['Ann. Vol']:>7.1%} {m['Sharpe']:>7.2f} {m['Sortino']:>7.2f} "
                  f"{m['Max DD']:>7.1%} {m.get('Skewness',0):>+6.2f}")

    # ================================================================
    # 2. Composite portfolio
    # ================================================================
    print(f"\n\n{'='*90}")
    print("COMPOSITE PORTFOLIO")
    print(f"{'='*90}")

    # ---- Equal Weight Composite ----
    composite_ret = pd.Series(0.0, index=ret_df.index)
    weights = pd.Series(0.0, index=ret_df.columns)

    for bucket_name, (codes, bucket_alloc) in ALLOC.items():
        available = [c for c in codes if c in ret_df.columns]
        if not available:
            continue
        per_strat_weight = bucket_alloc / len(available)
        print(f"\n  {bucket_name}: {bucket_alloc:.0%} total, {per_strat_weight:.2%} per strategy ({len(available)} strats)")
        for code in available:
            composite_ret += per_strat_weight * ret_df[code]
            weights[code] = per_strat_weight
            print(f"    {code}: {per_strat_weight:.2%}")

    composite_perf = (1 + composite_ret).cumprod() * 100
    composite_perf.iloc[0] = 100

    cm = compute_metrics(composite_perf)
    print(f"\n  EQUAL-WEIGHT COMPOSITE METRICS:")
    pct_keys = {"Total Return", "Ann. Return", "Ann. Vol", "Max DD", "Win Rate"}
    for k, v in cm.items():
        if k == "Days":
            print(f"    {k:20s} {int(v)}")
        elif k in pct_keys:
            print(f"    {k:20s} {v:>+.2%}")
        elif isinstance(v, float):
            print(f"    {k:20s} {v:>+.4f}")

    # ---- Optimized Composite: Sharpe-weighted carries, risk-parity hedges ----
    print(f"\n  {'-'*60}")
    print(f"  OPTIMIZED COMPOSITE (Sharpe-weighted carries, risk-parity hedges)")
    print(f"  {'-'*60}")

    opt_weights = pd.Series(0.0, index=ret_df.columns)

    # Carry: weight = max(Sharpe, 0)^1.5 (power > 1 concentrates on winners)
    carry_available = [c for c in CARRY if c in ret_df.columns]
    carry_sharpes = {}
    for c in carry_available:
        m = metrics_all.get(c, {})
        carry_sharpes[c] = max(m.get("Sharpe", 0), 0.0) ** 1.5

    carry_total_score = sum(carry_sharpes.values())
    if carry_total_score > 0:
        for c in carry_available:
            opt_weights[c] = 0.60 * carry_sharpes[c] / carry_total_score

    # Hedges: inverse-vol weighting (60% to hedges since they're insurance)
    hedge_available = [c for c in HEDGE if c in ret_df.columns]
    hedge_vols = {}
    for c in hedge_available:
        m = metrics_all.get(c, {})
        vol = m.get("Ann. Vol", 0.10)
        # Only include hedges with bear Sharpe > -0.5 or positive overall Sharpe
        sr = m.get("Sharpe", 0)
        hedge_vols[c] = 1.0 / max(vol, 0.01) if sr > -0.40 else 0.0

    hedge_total_inv_vol = sum(hedge_vols.values())
    if hedge_total_inv_vol > 0:
        for c in hedge_available:
            opt_weights[c] = 0.40 * hedge_vols[c] / hedge_total_inv_vol

    # Normalize to sum to 1
    if opt_weights.sum() > 0:
        opt_weights = opt_weights / opt_weights.sum()

    for code in carry_available + hedge_available:
        if opt_weights[code] > 0.001:
            cat = "carry" if code in CARRY else "hedge"
            print(f"    {code} ({cat}): {opt_weights[code]:.1%}")

    opt_composite_ret = (ret_df * opt_weights).sum(axis=1)
    opt_composite_perf = (1 + opt_composite_ret).cumprod() * 100
    opt_composite_perf.iloc[0] = 100

    opt_cm = compute_metrics(opt_composite_perf)
    print(f"\n  OPTIMIZED COMPOSITE METRICS:")
    for k, v in opt_cm.items():
        if k == "Days":
            print(f"    {k:20s} {int(v)}")
        elif k in pct_keys:
            print(f"    {k:20s} {v:>+.2%}")
        elif isinstance(v, float):
            print(f"    {k:20s} {v:>+.4f}")

    # ================================================================
    # 3. Diversification quality
    # ================================================================
    print(f"\n\n{'='*90}")
    print("DIVERSIFICATION QUALITY")
    print(f"{'='*90}")

    # Correlation matrix
    corr = ret_df.corr()
    carry_codes = [c for c in CARRY if c in ret_df.columns]
    hedge_codes = [c for c in HEDGE if c in ret_df.columns]

    # Carry-Carry correlation
    if len(carry_codes) > 1:
        cc = corr.loc[carry_codes, carry_codes].values.copy()
        np.fill_diagonal(cc, np.nan)
        print(f"\n  Carry-Carry avg correlation:  {np.nanmean(cc):+.3f}")
        print(f"  Carry-Carry max correlation:  {np.nanmax(cc):+.3f}")

    # Hedge-Hedge correlation
    if len(hedge_codes) > 1:
        hh = corr.loc[hedge_codes, hedge_codes].values.copy()
        np.fill_diagonal(hh, np.nan)
        print(f"  Hedge-Hedge avg correlation:  {np.nanmean(hh):+.3f}")

    # Carry-Hedge correlation
    if carry_codes and hedge_codes:
        ch = corr.loc[carry_codes, hedge_codes].values
        print(f"  Carry-Hedge avg correlation:  {np.nanmean(ch):+.3f}")

    # All-pairs
    all_corr = corr.values.copy()
    np.fill_diagonal(all_corr, np.nan)
    print(f"  All pairs avg correlation:    {np.nanmean(all_corr):+.3f}")
    print(f"  All pairs max correlation:    {np.nanmax(all_corr):+.3f}")

    # Effective N (PCA)
    eff_n = compute_effective_n(ret_df)
    print(f"\n  Effective N (PCA):            {eff_n:.1f}  (out of {len(ret_df.columns)} strategies)")

    # Diversification ratio
    div_ratio = compute_diversification_ratio(ret_df, weights)
    print(f"  Diversification Ratio:        {div_ratio:.2f}")

    # Top/bottom correlation pairs
    print(f"\n  Top 5 most correlated pairs:")
    pairs = []
    codes_list = list(ret_df.columns)
    for i in range(len(codes_list)):
        for j in range(i + 1, len(codes_list)):
            pairs.append((codes_list[i], codes_list[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, r in pairs[:5]:
        print(f"    {a:8s} / {b:8s}  {r:+.3f}")

    print(f"\n  5 least correlated pairs:")
    pairs.sort(key=lambda x: abs(x[2]))
    for a, b, r in pairs[:5]:
        print(f"    {a:8s} / {b:8s}  {r:+.3f}")

    # ================================================================
    # 4. Stress behavior
    # ================================================================
    print(f"\n\n{'='*90}")
    print("STRESS BEHAVIOR")
    print(f"{'='*90}")

    stress_corr, calm_corr = compute_conditional_correlation(ret_df)
    print(f"\n  Conditional correlation (bottom 10% days):  {stress_corr:+.3f}")
    print(f"  Conditional correlation (other 90% days):   {calm_corr:+.3f}")
    print(f"  Stress-Calm spread:                         {stress_corr - calm_corr:+.3f}")

    spot = load_spot(symbol)
    if not spot.empty:
        spot = spot.reindex(perf_df.index).ffill().dropna()
        if len(spot) > 100:
            crisis_beta = compute_crisis_beta(ret_df, spot, threshold=-0.01)
            print(f"  Crisis beta ({symbol} < -1% days):               {crisis_beta:+.3f}" if np.isfinite(crisis_beta) else "  Crisis beta: N/A")

            # Carry-specific crisis beta
            if carry_codes:
                carry_beta = compute_crisis_beta(ret_df[carry_codes], spot, threshold=-0.01)
                print(f"  Carry crisis beta:                          {carry_beta:+.3f}" if np.isfinite(carry_beta) else "  Carry crisis beta: N/A")
            if hedge_codes:
                hedge_beta = compute_crisis_beta(ret_df[hedge_codes], spot, threshold=-0.01)
                print(f"  Hedge crisis beta:                          {hedge_beta:+.3f}" if np.isfinite(hedge_beta) else "  Hedge crisis beta: N/A")

    # ================================================================
    # 5. Regime robustness
    # ================================================================
    print(f"\n\n{'='*90}")
    print("REGIME ROBUSTNESS")
    print(f"{'='*90}")

    if not spot.empty and len(spot) > 200:
        bull_sr, bear_sr, range_sr = compute_regime_sharpes(composite_perf, spot)
        print(f"\n  Composite Sharpe in Bull regime:   {bull_sr:+.2f}" if np.isfinite(bull_sr) else "  Bull: N/A")
        print(f"  Composite Sharpe in Bear regime:   {bear_sr:+.2f}" if np.isfinite(bear_sr) else "  Bear: N/A")
        print(f"  Composite Sharpe in Range regime:  {range_sr:+.2f}" if np.isfinite(range_sr) else "  Range: N/A")

        # Per-strategy regime Sharpes
        print(f"\n  Per-strategy Bear regime Sharpe:")
        for code in carry_codes + hedge_codes:
            if code in all_perfs:
                _, bear, _ = compute_regime_sharpes(all_perfs[code], spot)
                if np.isfinite(bear):
                    print(f"    {code:8s}  Bear Sharpe: {bear:+.2f}")

    # ================================================================
    # 6. Yearly returns
    # ================================================================
    print(f"\n\n{'='*90}")
    print("YEARLY RETURNS")
    print(f"{'='*90}")

    # Add both composites to ret_df for yearly analysis
    ret_with_composite = ret_df.copy()
    ret_with_composite["EQ_WT"] = composite_ret
    ret_with_composite["OPT_WT"] = opt_composite_ret
    yearly = compute_yearly_returns(ret_with_composite)
    print(yearly.to_string(float_format=lambda x: f"{x:+.1%}"))

    # ================================================================
    # 7. Marginal Sharpe contribution
    # ================================================================
    print(f"\n\n{'='*90}")
    print("MARGINAL SHARPE CONTRIBUTION (Sharpe_full - Sharpe_without)")
    print(f"{'='*90}")

    marginal = compute_marginal_sharpe(ret_df, weights)
    for code, val in sorted(marginal.items(), key=lambda x: x[1], reverse=True):
        cat = "carry" if code in CARRY else "hedge"
        print(f"  {code:8s} ({cat:6s}): {val:+.4f}")

    # ================================================================
    # 8. QQQ Benchmark
    # ================================================================
    if not spot.empty:
        spot_perf = spot.reindex(perf_df.index).ffill().dropna()
        if len(spot_perf) > 10:
            qm = compute_metrics(spot_perf / spot_perf.iloc[0] * 100)
            print(f"\n\n{'='*90}")
            print(f"{symbol} BUY & HOLD BENCHMARK")
            print(f"{'='*90}")
            print(f"  Ann. Return: {qm['Ann. Return']:>+.2%}   Vol: {qm['Ann. Vol']:>.2%}   "
                  f"Sharpe: {qm['Sharpe']:>.2f}   MaxDD: {qm['Max DD']:>+.2%}")

    # ================================================================
    # 9. Success criteria check
    # ================================================================
    print(f"\n\n{'='*90}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*90}")

    checks = {
        "Effective N > 5": eff_n > 5,
        "Avg carry-carry corr < 0.25": np.nanmean(cc) < 0.25 if len(carry_codes) > 1 else False,
        "Max pairwise corr < 0.70": np.nanmax(all_corr) < 0.70,
        "EqWt Composite Sharpe > 0.5": cm.get("Sharpe", 0) > 0.5,
        "OptWt Composite Sharpe > 0.5": opt_cm.get("Sharpe", 0) > 0.5,
        "Composite MaxDD > -20%": opt_cm.get("Max DD", -1) > -0.20,
        "Stress corr < calm corr": stress_corr < calm_corr,
    }
    for desc, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    # ================================================================
    # Export
    # ================================================================
    prefix = "opti_ptf_metrics" if opti else "portfolio_metrics"
    out_path = OUTPUT_DIR / f"{prefix}_{symbol}.xlsx"
    tmp_path = OUTPUT_DIR / f"{prefix}_{symbol}_tmp.xlsx"
    write_path = tmp_path if out_path.exists() else out_path
    try:
        write_path = out_path
        # Test write access by creating and removing a temp file
        out_path.touch()
        out_path.unlink()
    except (PermissionError, OSError):
        write_path = tmp_path
        print(f"  [WARN] {out_path} is locked, writing to {tmp_path}")
    with pd.ExcelWriter(write_path, engine="openpyxl") as writer:
        # Perf curves
        export_perf = perf_df.copy()
        export_perf["EQ_WT"] = composite_perf
        export_perf["OPT_WT"] = opt_composite_perf
        export_perf.to_excel(writer, sheet_name="PERF")

        # Daily returns
        ret_export = ret_df.copy()
        ret_export["EQ_WT"] = composite_ret
        ret_export["OPT_WT"] = opt_composite_ret
        ret_export.to_excel(writer, sheet_name="RETURNS")

        # Metrics
        metrics_rows = []
        for code, m in metrics_all.items():
            cat = "carry" if code in CARRY else "hedge"
            row = {"Code": code, "Category": cat}
            row.update(m)
            metrics_rows.append(row)
        cm_row = {"Code": "EQ_WT", "Category": "composite"}
        cm_row.update(cm)
        metrics_rows.append(cm_row)
        opt_row = {"Code": "OPT_WT", "Category": "composite"}
        opt_row.update(opt_cm)
        metrics_rows.append(opt_row)
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="METRICS", index=False)

        # Correlation
        corr.to_excel(writer, sheet_name="CORRELATION")

        # Yearly
        yearly.to_excel(writer, sheet_name="YEARLY")

        # Weights
        weights_df = pd.DataFrame({
            "Code": weights.index,
            "Category": ["carry" if c in CARRY else "hedge" for c in weights.index],
            "EqWt": weights.values,
            "OptWt": [opt_weights[c] for c in weights.index],
        })
        weights_df.to_excel(writer, sheet_name="WEIGHTS", index=False)

    print(f"\n\nExported to: {write_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flagship Portfolio Metrics")
    parser.add_argument("--version", type=str, default="4",
                        choices=["2", "3", "4", "5", "6", "7", "cboe"],
                        help="Suite version (2, 3, 4, 5, 6, 7, or cboe)")
    parser.add_argument("--symbol", type=str, default="QQQ",
                        help="Underlying symbol (QQQ, SPY, etc.)")
    parser.add_argument("--opti", action="store_true",
                        help="Use pruned optimal strategy subset (V6 only)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date filter, e.g. 2020-01-01")
    args = parser.parse_args()

    ver = args.version
    if ver == "2":
        OUTPUT_DIR = Path("outputs/flagship_suite_v2")
        FILE_PREFIX = "FSv2"
    elif ver == "3":
        OUTPUT_DIR = Path("outputs/flagship_suite_v3")
        FILE_PREFIX = "FS"
    elif ver == "5":
        OUTPUT_DIR = Path("outputs/flagship_suite_v5")
        FILE_PREFIX = "FS"
        CARRY = ["HVRP", "NKPT", "SDPS", "DVRP", "IRON", "PRSK", "V9PS",
                 "VXPS", "VXDH", "OTMC", "OMDH"]
        HEDGE = ["VLNG", "TAIL", "XHGE", "RCAL"]
        ALLOC["carry"] = (CARRY, 0.65)
        ALLOC["hedge"] = (HEDGE, 0.35)
    elif ver == "6":
        OUTPUT_DIR = Path("outputs/flagship_suite_v6")
        FILE_PREFIX = "FS"
        if args.opti:
            CARRY = ["VSS3", "VXDH", "OMDH", "SDPS"]
            HEDGE = ["THTA", "DVAR", "DVAS", "XHGE"]
        else:
            CARRY = ["VSWA", "VSUB", "VSS1", "VSS2", "VSS3",
                     "VXPS", "VXDH", "OTMC", "OMDH", "SDPS", "SDPL"]
            HEDGE = ["THTA", "THT2", "THT3", "DVAR", "DVAS", "XHGE", "XHGB", "XHGC"]
        ALLOC["carry"] = (CARRY, 0.60)
        ALLOC["hedge"] = (HEDGE, 0.40)
    elif ver == "7":
        OUTPUT_DIR = Path("outputs/flagship_final")
        FILE_PREFIX = "FS"
        CARRY = ["VSS3", "VXS3", "VCBA", "OMDH", "SDPS"]
        HEDGE = ["TH2L", "DVMX", "XHGE"]
        ALLOC["carry"] = (CARRY, 0.60)
        ALLOC["hedge"] = (HEDGE, 0.40)
    elif ver == "cboe":
        OUTPUT_DIR = Path("outputs/flagship_suite_cboe")
        FILE_PREFIX = "FS"
        CARRY = ["VSS3", "VXS3", "VCBA", "OMDH", "SDPS"]
        HEDGE = ["TH2L", "DVMX", "XHGE"]
        ALLOC["carry"] = (CARRY, 0.60)
        ALLOC["hedge"] = (HEDGE, 0.40)
    else:
        OUTPUT_DIR = Path("outputs/flagship_suite_v4")
        FILE_PREFIX = "FS"
        # V4 strategy codes (6 strategies: 3 carry + 3 hedge)
        CARRY = ["HVRP", "NKPT", "SDPS"]
        HEDGE = ["VLNG", "TAIL", "XHGE"]
        ALLOC["carry"] = (CARRY, 0.65)
        ALLOC["hedge"] = (HEDGE, 0.35)

    main(symbol=args.symbol, opti=args.opti, start_date=args.start)
