"""
Base Signal Engine — compute trading signals from market data.

Signals computed:
    From VIX/VVIX (market_data/ parquets, synthetic proxies):
        - vix, vix_zscore, vix_zscore_long, vix_pctl
        - vvix, vvix_zscore
        - vrp (variance risk premium = VIX^2 - RV^2), vrp_zscore

    From CBOE indices (market_data/ parquets, V3):
        - vix_cboe, vix3m, vix9d, vix_ts_ratio (VIX/VIX3M)
        - skew, skew_zscore

    From FRED macro data (market_data/ parquets, V3):
        - hy_oas, hy_oas_zscore, hy_oas_chg20d (HY credit spread)
        - yield_curve, yield_curve_zscore, yield_curve_chg20d (10Y-2Y)

    From spot data:
        - rv_5d, rv_10d, rv_20d, rv_60d (realized vol)
        - rv_ratio (short/long RV momentum)
        - ret_5d, ret_20d (spot return momentum)
        - sma_200 (200-day simple moving average)

    From options chain (computed on-the-fly in strategies):
        - IV term structure slope (near vs far ATM IV)
        - Put-call skew (25D put IV / ATM IV)
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _load_parquet_series(path: str, col: str) -> pd.Series:
    """Load a single column from a parquet file."""
    p = Path(path)
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index).normalize()
    if col in df.columns:
        return df[col].astype(float)
    if len(df.columns) > 0:
        return df.iloc[:, 0].astype(float)
    return pd.Series(dtype=float)


def compute_realized_vol(spot_series: pd.Series, window: int = 20) -> pd.Series:
    """Annualized close-to-close realized volatility."""
    returns = spot_series.pct_change()
    return returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def compute_rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of a series."""
    mean = series.rolling(window, min_periods=max(10, window // 3)).mean()
    std = series.rolling(window, min_periods=max(10, window // 3)).std()
    return (series - mean) / std.replace(0, np.nan)


def compute_rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank (0 to 1)."""
    return series.rolling(window, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


class BaseSignalEngine:
    """Builds a per-symbol DataFrame of daily signals."""

    def __init__(
        self,
        vix_path: str = "market_data/VIX.parquet",
        vvix_path: str = "market_data/VVIX.parquet",
        # V3 data paths
        vix_cboe_path: str = "market_data/VIX_CBOE.parquet",
        vix3m_path: str = "market_data/VIX3M.parquet",
        vix9d_path: str = "market_data/VIX9D.parquet",
        skew_path: str = "market_data/SKEW.parquet",
        hy_oas_path: str = "market_data/HY_OAS.parquet",
        t10y2y_path: str = "market_data/T10Y2Y.parquet",
        # V3+ additional data
        ovx_path: str = "market_data/OVX.parquet",
        gld_path: str = "market_data/GLD.parquet",
        move_path: str = "market_data/MOVE.parquet",
        pcr_path: str = "market_data/PCR_QQQ.parquet",
    ):
        # V2 synthetic signals
        self.vix = _load_parquet_series(vix_path, "vix")
        self.vvix = _load_parquet_series(vvix_path, "vvix")

        # V3 CBOE indices
        self.vix_cboe = _load_parquet_series(vix_cboe_path, "vix")
        self.vix3m = _load_parquet_series(vix3m_path, "vix3m")
        self.vix9d = _load_parquet_series(vix9d_path, "vix9d")
        self.skew = _load_parquet_series(skew_path, "skew")

        # V3 FRED macro
        self.hy_oas = _load_parquet_series(hy_oas_path, "hy_oas")
        self.t10y2y = _load_parquet_series(t10y2y_path, "yield_curve")

        # V3+ additional market data
        self.ovx = _load_parquet_series(ovx_path, "ovx")
        self.gld = _load_parquet_series(gld_path, "gld")
        self.move = _load_parquet_series(move_path, "move")
        self.pcr = _load_parquet_series(pcr_path, "pcr")

        if not self.vix.empty:
            print(f"[SIG] VIX loaded: {len(self.vix)} rows")
        else:
            print("[SIG] VIX data not found — VIX-based signals disabled")

        if not self.vvix.empty:
            print(f"[SIG] VVIX loaded: {len(self.vvix)} rows")
        else:
            print("[SIG] VVIX data not found — VVIX-based signals disabled")

        # V3 data summary
        v3_sources = {
            "VIX_CBOE": self.vix_cboe, "VIX3M": self.vix3m,
            "VIX9D": self.vix9d, "SKEW": self.skew,
            "HY_OAS": self.hy_oas, "T10Y2Y": self.t10y2y,
        }
        loaded = [k for k, v in v3_sources.items() if not v.empty]
        if loaded:
            print(f"[SIG] V3 data loaded: {', '.join(loaded)}")

    def build_signal_df(self, spot_series: pd.Series) -> pd.DataFrame:
        """Build a full signal DataFrame aligned to the spot_series index."""
        idx = spot_series.index
        sig = pd.DataFrame(index=idx)

        # --- Realized vol ---
        for w in [5, 10, 20, 60]:
            sig[f"rv_{w}d"] = compute_realized_vol(spot_series, w)

        # --- VIX signals ---
        if not self.vix.empty:
            vix_al = self.vix.reindex(idx, method="ffill")
            sig["vix"] = vix_al
            sig["vix_zscore"] = compute_rolling_zscore(vix_al, 60)
            sig["vix_zscore_long"] = compute_rolling_zscore(vix_al, 252)
            sig["vix_pctl"] = compute_rolling_percentile(vix_al, 252)
            sig["vix_ma_ratio"] = vix_al / vix_al.rolling(20, min_periods=10).mean()

        # --- VVIX signals ---
        if not self.vvix.empty:
            vvix_al = self.vvix.reindex(idx, method="ffill")
            sig["vvix"] = vvix_al
            sig["vvix_zscore"] = compute_rolling_zscore(vvix_al, 60)

        # --- VRP: Variance Risk Premium ---
        if "vix" in sig.columns:
            iv_var = (sig["vix"] / 100.0) ** 2  # VIX is in % → decimal variance
            rv_var = sig["rv_20d"] ** 2
            sig["vrp"] = iv_var - rv_var
            sig["vrp_zscore"] = compute_rolling_zscore(sig["vrp"], 60)

        # --- Spot momentum ---
        sig["ret_5d"] = spot_series.pct_change(5)
        sig["ret_20d"] = spot_series.pct_change(20)

        # --- RV momentum (short/long ratio) ---
        sig["rv_ratio"] = sig["rv_20d"] / sig["rv_60d"].replace(0, np.nan)

        # --- V3: CBOE VIX term structure ---
        if not self.vix_cboe.empty:
            vix_cboe_al = self.vix_cboe.reindex(idx, method="ffill")
            sig["vix_cboe"] = vix_cboe_al
        if not self.vix3m.empty:
            vix3m_al = self.vix3m.reindex(idx, method="ffill")
            sig["vix3m"] = vix3m_al
        if not self.vix9d.empty:
            sig["vix9d"] = self.vix9d.reindex(idx, method="ffill")
        if "vix_cboe" in sig.columns and "vix3m" in sig.columns:
            sig["vix_ts_ratio"] = sig["vix_cboe"] / sig["vix3m"].replace(0, np.nan)

        # --- V3: CBOE SKEW ---
        if not self.skew.empty:
            skew_al = self.skew.reindex(idx, method="ffill")
            sig["skew"] = skew_al
            sig["skew_zscore"] = compute_rolling_zscore(skew_al, 60)

        # --- V3: HY credit spread ---
        if not self.hy_oas.empty:
            hy_al = self.hy_oas.reindex(idx, method="ffill")
            sig["hy_oas"] = hy_al
            sig["hy_oas_zscore"] = compute_rolling_zscore(hy_al, 60)
            sig["hy_oas_chg20d"] = hy_al.diff(20)

        # --- V3: Yield curve (10Y-2Y) ---
        if not self.t10y2y.empty:
            yc_al = self.t10y2y.reindex(idx, method="ffill")
            sig["yield_curve"] = yc_al
            sig["yield_curve_zscore"] = compute_rolling_zscore(yc_al, 60)
            sig["yield_curve_chg20d"] = yc_al.diff(20)

        # --- V3: 200-day SMA ---
        sig["sma_200"] = spot_series.rolling(200, min_periods=100).mean()

        # --- V3+: OVX (oil vol) ---
        if not self.ovx.empty:
            ovx_al = self.ovx.reindex(idx, method="ffill")
            sig["ovx"] = ovx_al
            sig["ovx_z"] = compute_rolling_zscore(ovx_al, 60)

        # --- V3+: GLD (gold, safe haven) ---
        if not self.gld.empty:
            gld_al = self.gld.reindex(idx, method="ffill")
            sig["gld"] = gld_al
            sig["gld_mom_20d"] = gld_al.pct_change(20)

        # --- V3+: MOVE (bond vol) ---
        if not self.move.empty:
            move_al = self.move.reindex(idx, method="ffill")
            sig["move"] = move_al
            sig["move_z"] = compute_rolling_zscore(move_al, 60)

        # --- V3+: Put-Call Ratio from chain ---
        if not self.pcr.empty:
            pcr_al = self.pcr.reindex(idx, method="ffill")
            sig["pcr"] = pcr_al
            sig["pcr_z"] = compute_rolling_zscore(pcr_al, 60)

        # --- V3+: RVX/VIX ratio (if RVX available) ---
        # Not available from Yahoo, skip for now

        return sig


# ================================================================
# Chain-based signals (computed on-the-fly from daily chain)
# ================================================================


def compute_iv_term_structure(chain: pd.DataFrame, spot: float) -> dict:
    """
    Compute IV term structure from an options chain.

    Returns dict with keys: near_iv, far_iv, slope.
    slope = (far_iv - near_iv) / near_iv  (positive = contango).
    """
    result = {"near_iv": np.nan, "far_iv": np.nan, "slope": np.nan}

    if chain is None or chain.empty:
        return result
    if "implied_volatility" not in chain.columns:
        return result
    if "dte" not in chain.columns:
        return result

    # ATM filter: moneyness 0.97–1.03
    ch = chain.copy()
    if "strike" in ch.columns:
        ch["_mny"] = ch["strike"].astype(float) / spot
        atm = ch[(ch["_mny"] > 0.97) & (ch["_mny"] < 1.03)]
    else:
        atm = ch
    if atm.empty:
        return result

    iv = atm["implied_volatility"].astype(float)

    # Near-term: 5–15 DTE
    near = iv[atm["dte"].between(5, 15)]
    result["near_iv"] = float(near.median()) if not near.empty else np.nan

    # Far-term: 25–45 DTE
    far = iv[atm["dte"].between(25, 45)]
    result["far_iv"] = float(far.median()) if not far.empty else np.nan

    if np.isfinite(result["near_iv"]) and np.isfinite(result["far_iv"]) and result["near_iv"] > 0:
        result["slope"] = (result["far_iv"] - result["near_iv"]) / result["near_iv"]

    return result


def compute_skew(chain: pd.DataFrame, spot: float) -> float:
    """
    Compute put-call IV skew: 25D Put IV / ATM IV.

    Values > 1 mean puts are expensive relative to ATM (steep skew).
    """
    if chain is None or chain.empty:
        return np.nan
    if "delta" not in chain.columns or "implied_volatility" not in chain.columns:
        return np.nan
    if "dte" not in chain.columns:
        return np.nan

    # Use 20–40 DTE options
    sub = chain[(chain["dte"] >= 20) & (chain["dte"] <= 40)]
    if sub.empty:
        sub = chain[(chain["dte"] >= 7) & (chain["dte"] <= 60)]
    if sub.empty:
        return np.nan

    iv = sub["implied_volatility"].astype(float)
    delta = sub["delta"].astype(float)

    # ATM IV: calls with delta ~0.50
    atm_mask = (sub["type"] == "C") & delta.between(0.40, 0.60)
    atm_iv = float(iv[atm_mask].median()) if atm_mask.any() else np.nan

    # 25D Put IV
    put_mask = (sub["type"] == "P") & delta.between(-0.30, -0.20)
    put_iv = float(iv[put_mask].median()) if put_mask.any() else np.nan

    if np.isfinite(atm_iv) and np.isfinite(put_iv) and atm_iv > 0:
        return put_iv / atm_iv
    return np.nan
