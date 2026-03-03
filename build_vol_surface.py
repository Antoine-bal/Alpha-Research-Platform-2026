"""
Build Vol Surface Cache — SVI parametric fitting per date × expiry.

Reads raw QQQ options chain, fits SVI (Stochastic Volatility Inspired) model
per (date, expiry) slice, extracts clean features, and outputs a single
parquet with one row per date.

SVI Model:
    w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))

    where k = log(K/F) is log-moneyness and w is total implied variance (IV² * T).
    Parameters: a (level), b (wing slope), ρ (tilt), m (shift), σ (curvature).

Output: cache_options/vol_surface_QQQ.parquet
    ~35 columns: SVI features at 4 tenors (7D, 14D, 30D, 60D), term structure,
    VRP, flow/positioning, z-scores.

No forward-looking bias: all rolling stats use backward-looking windows only.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

import argparse as _argparse

_SYMBOL = "QQQ"  # default; overridden by --symbol CLI arg

RAW_CHAIN_PATH = Path(f"alpha_options_raw/{_SYMBOL}.parquet")
RF_PATH = Path("risk_free_3m.parquet")
SPOT_PATH = Path(f"alpha_corp_actions/{_SYMBOL}_daily_adjusted.parquet")
OUTPUT_PATH = Path(f"cache_options/vol_surface_{_SYMBOL}.parquet")

# Target tenors for interpolation (days)
TARGET_DTES = [7, 14, 30, 60]


# ================================================================
# SVI Model
# ================================================================

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                       m: float, sigma: float) -> np.ndarray:
    """SVI total variance: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    km = k - m
    return a + b * (rho * km + np.sqrt(km ** 2 + sigma ** 2))


def svi_iv(k: np.ndarray, T: float, a: float, b: float, rho: float,
           m: float, sigma: float) -> np.ndarray:
    """Convert SVI total variance to implied vol."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-8)
    return np.sqrt(w / T)


def fit_svi(k: np.ndarray, w_market: np.ndarray, weights: np.ndarray,
            T: float) -> dict:
    """
    Fit SVI model to market total variance data.

    Parameters
    ----------
    k : log-moneyness = log(K/F)
    w_market : market total variance = IV^2 * T
    weights : fitting weights (1/vega or similar)
    T : time to expiry in years

    Returns
    -------
    dict with keys: a, b, rho, m, sigma, rmse, success
    """
    n = len(k)
    if n < 4:
        return {"success": False}

    # Normalize weights
    weights = weights / weights.sum()

    def objective(params):
        a, b, rho, m, sigma = params
        w_model = svi_total_variance(k, a, b, rho, m, sigma)
        residuals = (w_model - w_market) ** 2
        return float(np.sum(weights * residuals))

    # Initial guess from data
    w_atm = float(np.interp(0.0, k, w_market)) if len(k) > 1 else float(w_market.mean())
    w_atm = max(w_atm, 1e-4)

    x0 = [
        w_atm,       # a: ATM total variance
        0.1,         # b: wing slope
        -0.3,        # rho: slight negative skew
        0.0,         # m: centered at ATM
        0.1,         # sigma: curvature
    ]

    # Bounds: enforce no-arbitrage constraints
    bounds = [
        (-0.5, 2.0),     # a
        (1e-4, 5.0),     # b > 0
        (-0.99, 0.99),   # |rho| < 1
        (-0.5, 0.5),     # m
        (1e-4, 2.0),     # sigma > 0
    ]

    # Constraint: a + b*sigma*sqrt(1-rho^2) >= 0 (non-negative variance)
    def constraint_nonneg_var(params):
        a, b, rho, m, sigma = params
        return a + b * sigma * np.sqrt(1 - rho ** 2)

    constraints = [{"type": "ineq", "fun": constraint_nonneg_var}]

    try:
        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10}
        )

        if result.success or result.fun < 1e-3:
            a, b, rho, m, sigma = result.x
            w_fit = svi_total_variance(k, a, b, rho, m, sigma)
            rmse = float(np.sqrt(np.mean((w_fit - w_market) ** 2)))
            return {
                "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
                "rmse": rmse, "success": True, "n_points": n,
            }
    except Exception:
        pass

    # Fallback: try with different initial guesses
    for rho0 in [-0.5, -0.1, 0.0]:
        for b0 in [0.05, 0.2, 0.5]:
            x0_alt = [w_atm, b0, rho0, 0.0, 0.15]
            try:
                result = minimize(
                    objective, x0_alt, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-10}
                )
                if result.success or result.fun < 1e-3:
                    a, b, rho, m, sigma = result.x
                    w_fit = svi_total_variance(k, a, b, rho, m, sigma)
                    rmse = float(np.sqrt(np.mean((w_fit - w_market) ** 2)))
                    return {
                        "a": a, "b": b, "rho": rho, "m": m, "sigma": sigma,
                        "rmse": rmse, "success": True, "n_points": n,
                    }
            except Exception:
                continue

    return {"success": False}


# ================================================================
# Feature Extraction from SVI Parameters
# ================================================================

def extract_svi_features(params: dict, T: float) -> dict:
    """
    Extract standardized features from fitted SVI parameters.

    Returns dict with: atm_iv, skew, curvature, rho, wing_slope, min_var,
                       put_25d_iv, call_25d_iv
    """
    if not params.get("success", False):
        return {}

    a, b, rho, m, sigma = params["a"], params["b"], params["rho"], params["m"], params["sigma"]

    features = {}

    # ATM IV: IV at k=0 (forward moneyness)
    w_atm = svi_total_variance(np.array([0.0]), a, b, rho, m, sigma)[0]
    w_atm = max(w_atm, 1e-8)
    features["atm_iv"] = float(np.sqrt(w_atm / T))

    # Skew: dw/dk at k=0 normalized
    # dw/dk = b * (rho + (k-m) / sqrt((k-m)^2 + sigma^2))
    dw_dk_0 = b * (rho + (-m) / np.sqrt(m ** 2 + sigma ** 2))
    features["skew"] = float(dw_dk_0 / (2 * np.sqrt(w_atm))) if w_atm > 1e-8 else 0.0

    # Curvature: d²w/dk² at k=0
    # d²w/dk² = b * sigma² / ((k-m)² + sigma²)^(3/2)
    d2w_dk2_0 = b * sigma ** 2 / (m ** 2 + sigma ** 2) ** 1.5
    features["curvature"] = float(d2w_dk2_0)

    # Raw SVI parameters
    features["rho"] = float(rho)
    features["wing_slope"] = float(b)
    features["min_var"] = float(a + b * sigma * np.sqrt(1 - rho ** 2))

    # 25D put and call IVs (approximate log-moneyness for 25D)
    # For 25D put: k ≈ -atm_iv * sqrt(T) * 0.67 (from normal approx)
    # For 25D call: k ≈ +atm_iv * sqrt(T) * 0.67
    atm_iv = features["atm_iv"]
    k_25d = atm_iv * np.sqrt(T) * 0.67 if T > 0 else 0.05

    w_put = svi_total_variance(np.array([-k_25d]), a, b, rho, m, sigma)[0]
    w_call = svi_total_variance(np.array([k_25d]), a, b, rho, m, sigma)[0]
    features["put_25d_iv"] = float(np.sqrt(max(w_put, 1e-8) / T))
    features["call_25d_iv"] = float(np.sqrt(max(w_call, 1e-8) / T))

    # Fit quality
    features["rmse"] = params.get("rmse", np.nan)

    return features


# ================================================================
# Date-Level Processing
# ================================================================

def process_date(day_chain: pd.DataFrame, spot: float, rf_annual: float,
                 date: pd.Timestamp) -> dict:
    """
    Process one date's option chain:
    1. Group by expiry
    2. Fit SVI per expiry
    3. Interpolate to target DTEs
    4. Compute flow features

    Returns dict of features for this date.
    """
    features = {}

    # Compute DTE
    day_chain = day_chain.copy()
    day_chain["expiration_dt"] = pd.to_datetime(day_chain["expiration"])
    day_chain["dte"] = (day_chain["expiration_dt"] - date).dt.days

    # Filter: reasonable DTE range, bid > 0
    valid = day_chain[
        (day_chain["dte"] >= 2) &
        (day_chain["dte"] <= 90) &
        (day_chain["bid"] > 0) &
        (day_chain["ask"] > 0) &
        (day_chain["implied_volatility"] > 0.01) &
        (day_chain["implied_volatility"] < 3.0)
    ].copy()

    if valid.empty:
        return features

    # Spread filter: (ask-bid)/mid < 1.0
    valid["mid"] = (valid["bid"] + valid["ask"]) / 2
    valid["spread_pct"] = (valid["ask"] - valid["bid"]) / valid["mid"].replace(0, np.nan)
    valid = valid[valid["spread_pct"] < 1.0]

    if valid.empty:
        return features

    # Forward price per expiry
    valid["T"] = valid["dte"] / 365.0
    valid["forward"] = spot * np.exp(rf_annual * valid["T"])

    # Log-moneyness
    valid["k"] = np.log(valid["strike"] / valid["forward"])

    # Market total variance
    valid["w_market"] = valid["implied_volatility"] ** 2 * valid["T"]

    # Filter deep OTM/ITM: |delta| > 0.02
    if "delta" in valid.columns:
        valid = valid[valid["delta"].abs() > 0.02]

    # ---- Fit SVI per expiry ----
    svi_fits = {}  # dte → {params, T, features}

    for expiry, grp in valid.groupby("expiration"):
        if len(grp) < 5:
            continue

        T = grp["T"].iloc[0]
        if T < 1 / 365:
            continue

        k_arr = grp["k"].values.astype(float)
        w_arr = grp["w_market"].values.astype(float)

        # Weights: prefer ATM (higher vega, more liquid)
        # Use 1 / (1 + |k|) as proxy for vega weighting
        wts = 1.0 / (1.0 + np.abs(k_arr) * 3.0)

        # Sort by k for stability
        sort_idx = np.argsort(k_arr)
        k_arr = k_arr[sort_idx]
        w_arr = w_arr[sort_idx]
        wts = wts[sort_idx]

        dte = int(grp["dte"].iloc[0])
        params = fit_svi(k_arr, w_arr, wts, T)

        if params.get("success"):
            feat = extract_svi_features(params, T)
            svi_fits[dte] = {"params": params, "T": T, "features": feat}

    if not svi_fits:
        return features

    # ---- Interpolate to target DTEs ----
    available_dtes = sorted(svi_fits.keys())

    for target_dte in TARGET_DTES:
        prefix = f"svi_{target_dte}d"

        # Find nearest or interpolate
        feat = _interpolate_features(svi_fits, available_dtes, target_dte)
        if feat:
            for key, val in feat.items():
                features[f"{prefix}_{key}"] = val

    # ---- Term structure features ----
    iv_7 = features.get("svi_7d_atm_iv")
    iv_14 = features.get("svi_14d_atm_iv")
    iv_30 = features.get("svi_30d_atm_iv")
    iv_60 = features.get("svi_60d_atm_iv")

    if iv_7 and iv_30 and iv_7 > 0:
        features["ts_slope_7_30"] = (iv_30 - iv_7) / iv_7
    if iv_30 and iv_60 and iv_30 > 0:
        features["ts_slope_30_60"] = (iv_60 - iv_30) / iv_30
    if iv_7 and iv_60 and iv_7 > 0:
        features["ts_slope_7_60"] = (iv_60 - iv_7) / iv_7

    # ---- Flow / positioning features ----
    flow_chain = day_chain[
        (day_chain["dte"] >= 7) & (day_chain["dte"] <= 45)
    ]
    if not flow_chain.empty:
        puts = flow_chain[flow_chain["type"].str.upper().isin(["P", "PUT"])]
        calls = flow_chain[flow_chain["type"].str.upper().isin(["C", "CALL"])]

        # Put/Call volume ratio
        put_vol = puts["volume"].sum()
        call_vol = calls["volume"].sum()
        if call_vol > 0:
            features["pc_vol_ratio"] = float(put_vol / call_vol)

        # Put/Call OI ratio
        put_oi = puts["open_interest"].sum()
        call_oi = calls["open_interest"].sum()
        if call_oi > 0:
            features["pc_oi_ratio"] = float(put_oi / call_oi)

        # Gamma exposure (GEX): sum(gamma * OI * contract_multiplier * spot^2 / 100)
        # For calls: positive GEX, for puts: negative GEX (dealer hedging)
        if "gamma" in flow_chain.columns:
            call_gex = (calls["gamma"] * calls["open_interest"] * spot * 0.01).sum()
            put_gex = -(puts["gamma"] * puts["open_interest"] * spot * 0.01).sum()
            total_gex = call_gex + put_gex
            # Normalize by spot level
            features["gex"] = float(total_gex / spot) if spot > 0 else 0.0

    return features


def _interpolate_features(svi_fits: dict, available_dtes: list,
                          target_dte: int) -> dict:
    """Interpolate SVI features to target DTE from available expiry fits."""
    if not available_dtes:
        return {}

    # Exact match
    if target_dte in svi_fits:
        return svi_fits[target_dte]["features"]

    # Find bracketing DTEs
    lower = [d for d in available_dtes if d <= target_dte]
    upper = [d for d in available_dtes if d >= target_dte]

    if lower and upper:
        d_lo = lower[-1]
        d_up = upper[0]
        if d_lo == d_up:
            return svi_fits[d_lo]["features"]
        # Linear interpolation weight
        w_lo = (d_up - target_dte) / (d_up - d_lo)
        w_up = 1 - w_lo
        feat_lo = svi_fits[d_lo]["features"]
        feat_up = svi_fits[d_up]["features"]
        result = {}
        for key in feat_lo:
            if key in feat_up:
                result[key] = feat_lo[key] * w_lo + feat_up[key] * w_up
        return result
    elif lower:
        # Extrapolate from nearest lower
        return svi_fits[lower[-1]]["features"]
    elif upper:
        # Extrapolate from nearest upper
        return svi_fits[upper[0]]["features"]

    return {}


# ================================================================
# Time-Series Features (computed after all dates are processed)
# ================================================================

def add_timeseries_features(df: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    """Add time-series derived features: z-scores, percentiles, vol-of-vol, VRP."""
    out = df.copy()

    # ---- VRP features ----
    # RV from spot
    spot_al = spot.reindex(df.index, method="ffill")
    ret = spot_al.pct_change()

    for w in [7, 20, 30]:
        rv = ret.rolling(w, min_periods=max(3, w // 2)).std() * np.sqrt(252)
        out[f"rv_{w}d"] = rv

    # VRP = IV² - RV²
    for dte in [7, 30]:
        iv_col = f"svi_{dte}d_atm_iv"
        rv_col = f"rv_{dte}d"
        if iv_col in out.columns and rv_col in out.columns:
            out[f"vrp_{dte}d"] = out[iv_col] ** 2 - out[rv_col] ** 2

    if "vrp_7d" in out.columns and "vrp_30d" in out.columns:
        out["vrp_ts_slope"] = out["vrp_7d"] - out["vrp_30d"]

    # ---- Rolling z-scores (60-day) ----
    zscore_cols = [
        "svi_30d_atm_iv", "svi_30d_skew", "svi_30d_curvature",
        "gex", "pc_vol_ratio",
    ]
    for col in zscore_cols:
        if col in out.columns:
            s = out[col]
            mu = s.rolling(60, min_periods=20).mean()
            sd = s.rolling(60, min_periods=20).std()
            out[f"{col}_z60"] = (s - mu) / sd.replace(0, np.nan)

    # ---- Vol of Vol ----
    if "svi_30d_atm_iv" in out.columns:
        iv_chg = out["svi_30d_atm_iv"].diff()
        out["vol_of_vol_20d"] = iv_chg.rolling(20, min_periods=10).std() * np.sqrt(252)
        # Z-score of vol-of-vol
        vov = out["vol_of_vol_20d"]
        mu = vov.rolling(60, min_periods=20).mean()
        sd = vov.rolling(60, min_periods=20).std()
        out["vol_of_vol_z60"] = (vov - mu) / sd.replace(0, np.nan)

    # ---- IV percentiles ----
    if "svi_30d_atm_iv" in out.columns:
        iv = out["svi_30d_atm_iv"]
        out["iv_pctl_252d"] = iv.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        out["iv_pctl_126d"] = iv.rolling(126, min_periods=40).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    return out


# ================================================================
# Main
# ================================================================

def load_spot() -> pd.Series:
    """Load QQQ adjusted close."""
    df = pd.read_parquet(SPOT_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    for col in ["adjusted_close", "close", "spot"]:
        if col in df.columns:
            return df[col]
    return df.iloc[:, 0]


def load_rf() -> pd.DataFrame:
    """Load risk-free rates."""
    df = pd.read_parquet(RF_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df


def main():
    print("=" * 70)
    print(f"  SVI Vol Surface Builder — {_SYMBOL}")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1] Loading raw chain...")
    chain = pd.read_parquet(RAW_CHAIN_PATH)
    chain["date"] = pd.to_datetime(chain["date"])
    dates = sorted(chain["date"].unique())
    print(f"    {len(chain):,} rows, {len(dates)} dates")
    print(f"    Date range: {dates[0].date()} to {dates[-1].date()}")

    print("\n[2] Loading spot & risk-free rates...")
    spot = load_spot()
    rf = load_rf()
    print(f"    Spot: {len(spot)} days")
    print(f"    RF: {len(rf)} days")

    # ---- Process each date ----
    print(f"\n[3] Fitting SVI per date × expiry...")
    all_features = {}
    n_success = 0
    n_fail = 0

    for i, dt in enumerate(dates):
        if i % 100 == 0:
            print(f"    Processing date {i+1}/{len(dates)}: {dt.date()}  "
                  f"(fits: {n_success}, fails: {n_fail})")

        # Get spot and RF for this date
        spot_val = spot.asof(dt)
        if pd.isna(spot_val) or spot_val <= 0:
            continue

        rf_row = rf.index.asof(dt)
        rf_annual = float(rf.loc[rf_row, "rf_annual"]) if pd.notna(rf_row) else 0.05

        # Get chain for this date
        day_chain = chain[chain["date"] == dt]
        if day_chain.empty:
            continue

        features = process_date(day_chain, float(spot_val), rf_annual, dt)
        if features:
            all_features[dt] = features
            n_success += 1
        else:
            n_fail += 1

    print(f"\n    Done: {n_success} dates with features, {n_fail} failed")

    # ---- Build DataFrame ----
    print("\n[4] Building feature DataFrame...")
    df = pd.DataFrame.from_dict(all_features, orient="index")
    df.index.name = "date"
    df = df.sort_index()
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # ---- Add time-series features ----
    print("\n[5] Computing time-series features...")
    df = add_timeseries_features(df, spot)
    print(f"    Final shape: {df.shape}")

    # ---- Coverage stats ----
    print("\n[6] Feature coverage:")
    for col in sorted(df.columns):
        pct = df[col].notna().mean() * 100
        if pct > 0:
            print(f"    {col:35s}: {pct:5.1f}% non-null  "
                  f"[{df[col].min():.4f}, {df[col].max():.4f}]")

    # ---- Save ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    print(f"\n[7] Saved to {OUTPUT_PATH}")
    print(f"    {len(df)} rows × {len(df.columns)} columns")
    print("=" * 70)


if __name__ == "__main__":
    _parser = _argparse.ArgumentParser(description="SVI Vol Surface Builder")
    _parser.add_argument("--symbol", type=str, default="QQQ",
                         help="Underlying symbol (QQQ, SPY, etc.)")
    _args = _parser.parse_args()
    _SYMBOL = _args.symbol
    RAW_CHAIN_PATH = Path(f"alpha_options_raw/{_SYMBOL}.parquet")
    SPOT_PATH = Path(f"alpha_corp_actions/{_SYMBOL}_daily_adjusted.parquet")
    OUTPUT_PATH = Path(f"cache_options/vol_surface_{_SYMBOL}.parquet")
    main()
