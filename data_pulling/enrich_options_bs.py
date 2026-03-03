# enrich_options_bs.py

import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from bs_pricing import implied_vol_newton, bs_greeks

# ---------- Config ---------- #

SYMBOLS = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]

START_DATE = pd.Timestamp("2020-01-01")
END_DATE   = pd.Timestamp("2025-11-16")

OPTIONS_RAW_DIR      = pathlib.Path("alpha_options_raw")
OPTIONS_ENRICHED_DIR = pathlib.Path("alpha_options_enriched")
CORP_DIR             = pathlib.Path("alpha_corp_actions")
RISK_FREE_PATH       = pathlib.Path("risk_free_3m.parquet")

# Fallback risk-free rate if file missing (annualised, e.g. 0.03 = 3%)
R_FALLBACK = 0.03

MAX_ROWS_PER_CHUNK = 500_000  # chunking inside a symbol if huge


# ---------- Helpers ---------- #

def load_risk_free() -> pd.DataFrame:
    """
    Load daily 3M risk-free rate from risk_free_3m.parquet if present,
    else return constant series over [START_DATE, END_DATE].
    """
    if RISK_FREE_PATH.exists():
        df = pd.read_parquet(RISK_FREE_PATH)
        if "date" not in df.columns or "rf_annual" not in df.columns:
            raise ValueError("risk_free_3m.parquet must have ['date','rf_annual'] columns.")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
        df = df.sort_values("date")
        return df[["date", "rf_annual"]].reset_index(drop=True)
    else:
        dates = pd.date_range(START_DATE, END_DATE, freq="B")
        return pd.DataFrame({"date": dates, "rf_annual": R_FALLBACK})


def load_spot_for_symbol(sym: str) -> pd.DataFrame:
    """
    Load split-normalized spot AND price_factor for a symbol from alpha_corp_actions,
    using the SAME logic as MarketData._load_spot in the backtester.

    Returns DataFrame indexed by 'date' with columns:
        - 'spot'         : split-normalised close
        - 'price_factor' : level_last / split_level (used to normalize strikes)
    """
    path = CORP_DIR / f"{sym}_daily_adjusted.parquet"
    if not path.exists():
        print(f"[WARN] Spot corp file missing for {sym}: {path}")
        return pd.DataFrame(columns=["date", "spot", "price_factor"]).set_index("date")

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        print(f"[WARN] No 'date' column in corp file for {sym}, skipping.")
        return pd.DataFrame(columns=["date", "spot", "price_factor"]).set_index("date")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    if df.empty:
        print(f"[WARN] No spot rows for {sym} in backtest window (corp file).")
        return pd.DataFrame(columns=["date", "spot", "price_factor"]).set_index("date")

    df = df.sort_values("date").set_index("date")

    # Use raw close if available, else adj_close (same as backtester)
    if "close" in df.columns:
        close_raw = df["close"].astype(float)
    elif "adj_close" in df.columns:
        close_raw = df["adj_close"].astype(float)
    else:
        print(f"[WARN] {sym}: neither 'close' nor 'adj_close' in corp file.")
        return pd.DataFrame(columns=["date", "spot", "price_factor"]).set_index("date")

    # split-only series
    if "split_coefficient" in df.columns:
        split_raw = df["split_coefficient"].astype(float)
        split_raw = split_raw.replace(0.0, np.nan).fillna(1.0)
    else:
        split_raw = pd.Series(1.0, index=df.index)

    split_level = split_raw.cumprod()
    level_last = float(split_level.iloc[-1])
    if level_last <= 0:
        level_last = 1.0

    price_factor = level_last / split_level
    spot_norm = close_raw / price_factor

    out = pd.DataFrame(
        {
            "spot": spot_norm.astype(float),
            "price_factor": price_factor.astype(float),
        },
        index=df.index,
    )
    return out

# ---------- Main enrichment per symbol ---------- #

def enrich_symbol(sym: str, rf_df: pd.DataFrame):
    src_path = OPTIONS_RAW_DIR / f"{sym}.parquet"
    if not src_path.exists():
        print(f"[WARN] Raw options file missing for {sym}: {src_path}")
        return

    df = pd.read_parquet(src_path)
    if "date" not in df.columns or "expiration" not in df.columns:
        print(f"[WARN] {sym}: 'date' or 'expiration' missing, skipping.")
        return

    # Normalise dates and clip to backtest window
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    if df.empty:
        print(f"[INFO] {sym}: no rows in backtest window.")
        return

    df = df.sort_values(["date", "expiration"]).reset_index(drop=True)

    # Compute mid price (same logic as in backtester)
    if "mid" not in df.columns:
        if "mark" in df.columns:
            df["mid"] = df["mark"].astype(float)
        else:
            bid = df.get("bid", np.nan).astype(float)
            ask = df.get("ask", np.nan).astype(float)
            df["mid"] = np.where(
                np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0),
                0.5 * (bid + ask),
                df.get("last", np.nan).astype(float),
            )

    # Filter to options where bid > 0.5 & mid > 0 (for BS sanity)
    if "bid" in df.columns:
        bid = df["bid"].astype(float)
    else:
        bid = pd.Series(np.nan, index=df.index)

    mask_valid = (bid > 0.5) & (df["mid"] > 0)
    if not mask_valid.any():
        print(f"[WARN] {sym}: no rows with bid>0.5 & mid>0.")
        OPTIONS_ENRICHED_DIR.mkdir(exist_ok=True)
        df.to_parquet(OPTIONS_ENRICHED_DIR / f"{sym}.parquet", index=False)
        return

    # Merge risk-free
    rf_df_local = rf_df.copy()
    rf_df_local = rf_df_local.rename(columns={"date": "date_rf"})
    rf_df_local["date_rf"] = rf_df_local["date_rf"].dt.normalize()

    df["date_merge"] = df["date"]
    df = df.merge(
        rf_df_local,
        left_on="date_merge",
        right_on="date_rf",
        how="left",
    )
    df["rf_annual"] = df["rf_annual"].fillna(R_FALLBACK)
    df.drop(columns=["date_merge", "date_rf"], inplace=True)

    # Load split-normalized spot from corp files and merge
    spot_df = load_spot_for_symbol(sym)
    if spot_df.empty:
        print(f"[WARN] {sym}: no spot available; cannot compute BS greeks.")
        OPTIONS_ENRICHED_DIR.mkdir(exist_ok=True)
        df.to_parquet(OPTIONS_ENRICHED_DIR / f"{sym}.parquet", index=False)
        return

    spot_df = spot_df.reset_index().rename(columns={"date": "date_spot"})
    df = df.merge(
        spot_df,
        left_on="date",
        right_on="date_spot",
        how="left",
    )
    df.drop(columns=["date_spot"], inplace=True)

    if "spot" not in df.columns:
        print(f"[WARN] {sym}: 'spot' missing after merge; cannot compute BS greeks.")
        OPTIONS_ENRICHED_DIR.mkdir(exist_ok=True)
        df.to_parquet(OPTIONS_ENRICHED_DIR / f"{sym}.parquet", index=False)
        return

    # Time to maturity in years (ACT/365)
    df["T_years"] = (df["expiration"] - df["date"]).dt.days / 365.0

    # Underlying price for BS
    S = df["spot"].astype(float).values  # split-normalised spot

    # Ensure we have price_factor; if missing, fallback to 1.0 (no split adjustment)
    if "price_factor" in df.columns:
        price_factor = df["price_factor"].astype(float).replace(0.0, np.nan).fillna(1.0).values
    else:
        price_factor = np.ones(len(df), dtype=float)

    # Effective strike in the same split-adjusted space as spot
    K_eff = df["strike"].astype(float).values / price_factor

    K = K_eff
    T = df["T_years"].astype(float).values
    r = df["rf_annual"].astype(float).values
    mid = df["mid"].astype(float).values

    if "type" not in df.columns:
        print(f"[WARN] {sym}: 'type' missing, cannot determine call/put.")
        OPTIONS_ENRICHED_DIR.mkdir(exist_ok=True)
        df.to_parquet(OPTIONS_ENRICHED_DIR / f"{sym}.parquet", index=False)
        return

    opt_type = df["type"].astype(str).str.upper().str[0]
    is_call = (opt_type == "C").values

    # Prepare BS columns
    df["iv_BS"] = np.nan
    df["delta_BS"] = np.nan
    df["gamma_BS"] = np.nan
    df["vega_BS"] = np.nan
    df["theta_BS"] = np.nan

    idx_valid = np.where(mask_valid.values)[0]
    n_valid = len(idx_valid)

    for start in tqdm(
        range(0, n_valid, MAX_ROWS_PER_CHUNK),
        desc=f"{sym} BS enrich",
        unit="rows",
    ):
        end = min(start + MAX_ROWS_PER_CHUNK, n_valid)
        sl = idx_valid[start:end]

        S_slice = S[sl]
        K_slice = K[sl]
        T_slice = T[sl]
        r_slice = r[sl]
        mid_slice = mid[sl]
        is_call_slice = is_call[sl]

        # IV
        iv_slice = implied_vol_newton(
            price=mid_slice,
            S=S_slice,
            K=K_slice,
            T=T_slice,
            r=r_slice,
            is_call=is_call_slice,
            q=0.0,
        )

        # Greeks
        delta_slice, gamma_slice, vega_slice, theta_slice = bs_greeks(
            S_slice,
            K_slice,
            T_slice,
            r_slice,
            sigma=iv_slice,
            is_call=is_call_slice,
            q=0.0,
        )

        df.loc[df.index[sl], "iv_BS"] = iv_slice
        df.loc[df.index[sl], "delta_BS"] = delta_slice
        df.loc[df.index[sl], "gamma_BS"] = gamma_slice
        df.loc[df.index[sl], "vega_BS"] = vega_slice
        df.loc[df.index[sl], "theta_BS"] = theta_slice

    # --------------- QC filter vs provider greeks --------------- #

    # have_provider = all(
    #     c in df.columns
    #     for c in ["delta", "gamma", "theta"]
    # )
    #
    # if have_provider:
    #     eps = 1e-8
    #
    #     valid_qc = (
    #         np.isfinite(df["delta_BS"]) & np.isfinite(df["delta"])
    #         & np.isfinite(df["gamma_BS"]) & np.isfinite(df["gamma"])
    #         & np.isfinite(df["theta_BS"]) & np.isfinite(df["theta"])
    #     )
    #
    #     rel_delta = np.abs(df["delta_BS"] - df["delta"]) / (np.abs(df["delta"]) + eps)
    #     rel_gamma = np.abs(df["gamma_BS"] - df["gamma"]) / (np.abs(df["gamma"]) + eps)
    #     rel_theta = np.abs(df["theta_BS"] - df["theta"]) / (np.abs(df["theta"]) + eps)
    #
    #     ok_delta = (rel_delta <= 0.03)
    #     ok_gamma = (rel_gamma <= 0.03)
    #     ok_theta = (rel_theta <= 0.03)
    #
    #     greeks_ok = valid_qc & ok_delta & ok_gamma & ok_theta
    #
    #     before = len(df)
    #     df = df[greeks_ok].copy()
    #     after = len(df)
    #     print(f"[INFO] {sym}: QC filter kept {after}/{before} rows "
    #           f"({after / max(before,1):.1%}).")
    # else:
    #     print(f"[WARN] {sym}: provider greeks missing, skipping QC filter.")

    # --------------- Save enriched file --------------- #

    def _insert_after(cols, base, new):
        if base in cols and new in cols:
            cols = cols.copy()
            cols.remove(new)
            i = cols.index(base)
            cols.insert(i + 1, new)
        return cols

    cols = list(df.columns)
    cols = _insert_after(cols, "implied_volatility", "iv_BS")
    cols = _insert_after(cols, "delta", "delta_BS")
    cols = _insert_after(cols, "gamma", "gamma_BS")
    cols = _insert_after(cols, "vega",  "vega_BS")
    cols = _insert_after(cols, "theta", "theta_BS")
    df = df[cols]

    OPTIONS_ENRICHED_DIR.mkdir(exist_ok=True)
    out_path = OPTIONS_ENRICHED_DIR / f"{sym}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved enriched options for {sym}: {len(df)} rows -> {out_path}")


def main():
    rf_df = load_risk_free()
    for sym in SYMBOLS:
        enrich_symbol(sym, rf_df)


if __name__ == "__main__":
    main()
