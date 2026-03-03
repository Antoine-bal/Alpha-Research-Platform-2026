import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, statsmodels.api as sm
from datetime import datetime
from math import log, sqrt, exp, erf
from pandas.tseries.offsets import BDay
import pyarrow.dataset as ds
import pyarrow.compute as pc
from collections import defaultdict

# aapl_path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\alpha_options_raw\AAPL_options_2020-01-01_to_2025-11-16.parquet"
# df = pd.read_parquet(aapl_path)
# # ========= Preparation from raw data =========
# path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\exports\option_chain.parquet"
#
# dataset = ds.dataset(path, format="parquet")

#######################  Creating usable csv from raw data  #######################################
#
# path = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\exports\option_chain.parquet"
#
# tickers = [
#     "NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "BRK.B", "TSLA",
#     "JPM", "V", "LLY", "NFLX", "XOM", "MA", "WMT", "COST", "ORCL", "JNJ",
#     "HD", "PG", "ABBV", "BAC", "UNH",
#     "CRM", "ADBE", "PYPL", "AMD", "INTC", "CSCO", "MCD", "NKE", "WFC", "CVX",
#     "PEP", "KO", "DIS", "BA", "MRK", "MO", "IBM", "T", "GM", "CAT", "UPS", "DOW",
#     "PLTR", "TXN", "LIN", "AMAT"
# ]
# cols = [
#     "date", "act_symbol", "expiration", "strike",
#     "call_put", "bid", "ask", "vol", "delta",
#     "gamma", "theta", "vega", "rho"
# ]

# dataset = ds.dataset(path, format="parquet")
#
# filt = pc.field("act_symbol").isin(tickers)
#
# table = dataset.to_table(columns=cols, filter=filt)
# df_all = table.to_pandas()
#
# print("Final:", df_all.shape)
#
# df_all.to_csv('my_data.csv')

#######################  LTTLE ANALYSIS #######################################
# Existing setup
# UNIVERSE = [
#     "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
#     "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
#     "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
# ]
#
# cols = ["act_symbol", "date", "expiration"]
# filt = pc.field("act_symbol").isin(UNIVERSE)
#
# dataset = ds.dataset(path, format="parquet")
#
# # ⬇ keeps your original metric
# min_dte_per_symbol = defaultdict(lambda: None)
#
# # ⬇ NEW: for each symbol we track: min(# expiries on a given date)
# min_exp_count_per_symbol = defaultdict(lambda: None)
#
# for batch in dataset.to_batches(columns=cols, filter=filt):
#     df = batch.to_pandas()
#     if df.empty:
#         continue
#
#     df["date"] = pd.to_datetime(df["date"]).dt.normalize()
#     df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
#     df["DTE"] = (df["expiration"] - df["date"]).dt.days
#     df = df[df["DTE"] > 0]
#     df = df[df['date']>= pd.Timestamp('2024-01-01')]
#
#     # ----- Update DTE metric -----
#     grouped_dte = df.groupby("act_symbol")["DTE"].min()
#     for sym, dte_min in grouped_dte.items():
#         current = min_dte_per_symbol[sym]
#         if current is None or dte_min < current:
#             min_dte_per_symbol[sym] = dte_min
#
#     # ----- NEW: expiration count per day -----
#     # For each (symbol, date), count distinct expiries
#     grouped_exp = df.groupby(["act_symbol", "date"])["expiration"].nunique()
#
#     # grouped_exp looks like:
#     #   act_symbol  date        -> number_of_expiries
#     # We update the minima per symbol
#     for (sym, _day), exp_count in grouped_exp.items():
#         current = min_exp_count_per_symbol[sym]
#         if current is None or exp_count < current:
#             min_exp_count_per_symbol[sym] = exp_count
#
# # ----- Final join into a single DataFrame -----
# rows = []
# for sym in sorted(min_dte_per_symbol.keys()):
#     rows.append({
#         "Symbol": sym,
#         "Min_DTE": min_dte_per_symbol[sym],
#         "Min_ExpiryCount_PerDay": min_exp_count_per_symbol.get(sym)
#     })
#
# result_df = pd.DataFrame(rows).sort_values(["Min_ExpiryCount_PerDay", "Min_DTE"]).reset_index(drop=True)
#
# print(result_df)


# df_all = table.to_pandas()
#
# print("Final:", df_all.shape)
#
# df_all.to_csv('my_data.csv')

# ========= Config =========
SPOT_CSV = "spot_data.csv"
EARNINGS_CSV = "earnings.csv"

ALPHA_OPTIONS_DIR = "alpha_options_raw"
USE_ALPHA_OPTIONS = True   # set False to fall back to old CSV logic

# Moneyness filter
MNY_MIN, MNY_MAX = 0.70, 1.30

OPTIONS_DIR = "options"
OPTIONS_SINGLE_CSV = "big_options_clean.csv"
CACHE_DIR = "cache_iv"
OUT_EXCEL = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings\outputs\earnings_iv_analysis.xlsx"
os.makedirs(CACHE_DIR, exist_ok=True)
OUT_DIR = os.path.dirname(OUT_EXCEL)
if OUT_DIR:
    os.makedirs(OUT_DIR, exist_ok=True)

UNIVERSE_FULL = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT"
]
UNIVERSE = UNIVERSE_FULL  # Full 51-ticker production run

# UNIVERSE = [
#     "MSFT"
# ]

STUDY_START = pd.Timestamp("2020-01-01")
STUDY_END   = pd.Timestamp("2025-11-14")
PRE, POST = 10, 10
BASELINE_PRE_TD, BASELINE_POST_TD = 30, 10
MAX_ANCHOR_LAG_DAYS = 3
BASE_PRE_MAT_DAYS = 30
R_RATE, Q_DIV = 0.03, 0.0
USE_CACHE, CACHE_PREFIX = True, "first_project"
KEEP_OPT_DATA_IN_RAM = True
IV_SCALE_AUTO_THRESHOLD = 3.0

plt.rcParams.update({"figure.figsize": (10,5), "axes.grid": True, "font.size": 12})

# ========= Black–Scholes / IV =========
def _norm_pdf(x): return (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*x*x)
def _norm_cdf(x): return 0.5*(1.0+erf(x/np.sqrt(2.0)))
def _bs_price(S,K,T,r,q,sigma,is_call):
    if T <= 0: return max(0.0,S-K) if is_call else max(0.0,K-S)
    sigma = max(sigma,1e-8)
    d1 = (log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrt(T)); d2 = d1 - sigma*sqrt(T)
    return (S*exp(-q*T)*_norm_cdf(d1) - K*exp(-r*T)*_norm_cdf(d2)) if is_call else (K*exp(-r*T)*_norm_cdf(-d2) - S*exp(-q*T)*_norm_cdf(-d1))
def _vega(S,K,T,r,q,sigma):
    if T<=0 or sigma<=1e-8: return 0.0
    d1 = (log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrt(T))
    return S*exp(-q*T)*sqrt(T)*_norm_pdf(d1)
def _iv_from_price(target,S,K,T,is_call,r=R_RATE,q=Q_DIV,sigma0=0.3,tol=1e-6,max_iter=40):
    if T<=0 or target<=0 or S<=0 or K<=0: return np.nan
    intrinsic = max(0.0,(S*exp(-q*T)-K*exp(-r*T)) if is_call else (K*exp(-r*T)-S*exp(-q*T)))
    if target <= intrinsic+1e-10: return 0.01
    lo,hi = 1e-4,5.0; sigma = float(np.clip(sigma0,lo,hi))
    for _ in range(max_iter):
        px = _bs_price(S,K,T,r,q,sigma,is_call); diff = px-target
        if abs(diff)<tol: return float(sigma)
        v = _vega(S,K,T,r,q,sigma)
        if v<=1e-8 or not np.isfinite(v):
            hi,sigma = (sigma,0.5*(lo+sigma)) if diff>0 else (hi,0.5*(sigma+hi)); lo = lo if diff>0 else sigma
            continue
        sigma_new = sigma - diff/v
        if not np.isfinite(sigma_new) or sigma_new<=lo or sigma_new>=hi:
            hi,sigma = (sigma,0.5*(lo+sigma)) if diff>0 else (hi,0.5*(sigma+hi)); lo = lo if diff>0 else sigma
        else:
            sigma = sigma_new
    return float(sigma if np.isfinite(sigma) else np.nan)

def _normalize_iv_scale(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    med = s.median()
    if np.isfinite(med) and med > IV_SCALE_AUTO_THRESHOLD:
        return s / 100.0
    return s

# ========= Spot loader =========
def load_spot_close_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]); df["date"] = df["date"].dt.normalize()
    px = "adj_close" if "adj_close" in df.columns else ("close" if "close" in df.columns else None)
    if px is None: raise ValueError("spot_data.csv must contain 'close' or 'adj_close'.")
    return (df[["symbol","date",px]].rename(columns={px:"spot"})
            .dropna(subset=["spot"]).set_index(["symbol","date"]).sort_index())

# ========= ATM helper =========
def pick_atm_row(sub: pd.DataFrame):
    if sub.empty: return None
    if "moneyness" not in sub.columns: sub = sub.assign(moneyness=sub["strike"]/sub["spot"])
    sub = sub.assign(abs_mny=(1.0-sub["moneyness"]).abs())
    return sub.loc[sub["abs_mny"].idxmin()]

# ========= Options loader + IV =========
def _load_raw_options_for_ticker(tkr: str) -> pd.DataFrame:
    """
    Load raw options for one ticker.

    Priority:
    1) If USE_ALPHA_OPTIONS and alpha_option_raw/TICKER.parquet exists, use that.
    2) Else, fall back to your previous CSV logic.
    """

    # --- 1) Try alpha_option_raw parquet ---
    if USE_ALPHA_OPTIONS:
        alpha_path = os.path.join(ALPHA_OPTIONS_DIR, f"{tkr}.parquet")
        if os.path.exists(alpha_path):
            df = pd.read_parquet(alpha_path)
            if df.empty:
                print(f"[{tkr}] alpha_option_raw parquet is empty")
                return df

            # Expected columns from your description:
            # ['contractID','symbol','expiration','strike','type','last','mark',
            #  'bid','bid_size','ask','ask_size','volume','open_interest',
            #  'date','implied_volatility','delta','gamma','theta','vega','rho']

            # Standardize date / expiry
            if "date" not in df.columns:
                raise ValueError(f"[{tkr}] alpha_option_raw: missing 'date'")
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()

            if "expiration" not in df.columns:
                raise ValueError(f"[{tkr}] alpha_option_raw: missing 'expiration'")
            df["expiry"] = pd.to_datetime(df["expiration"]).dt.normalize()

            # strike
            if "strike" not in df.columns:
                raise ValueError(f"[{tkr}] alpha_option_raw: missing 'strike'")
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

            # call / put
            if "type" not in df.columns:
                raise ValueError(f"[{tkr}] alpha_option_raw: missing 'type' (call/put)")
            t = df["type"].astype(str).str.strip().str.upper()
            df["is_call"] = t.isin(["C", "CALL"])

            # mid price: prefer 'mark', then (bid+ask)/2, then 'last'
            price_cols = {c.lower(): c for c in df.columns}
            mid = None
            if "mark" in price_cols:
                mid = df[price_cols["mark"]]
            elif "bid" in price_cols and "ask" in price_cols:
                mid = 0.5 * (
                    pd.to_numeric(df[price_cols["bid"]], errors="coerce")
                    + pd.to_numeric(df[price_cols["ask"]], errors="coerce")
                )
            elif "last" in price_cols:
                mid = pd.to_numeric(df[price_cols["last"]], errors="coerce")
            else:
                raise ValueError(f"[{tkr}] alpha_option_raw: no usable price column (mark/bid+ask/last)")

            df["mid"] = mid.astype(float)
            df = df[df["mid"] > 0].copy()

            return df

    # --- 2) Fallback: your previous CSV logic ---
    if OPTIONS_SINGLE_CSV and os.path.exists(OPTIONS_SINGLE_CSV):
        df = pd.read_csv(OPTIONS_SINGLE_CSV)
        if "symbol" not in df.columns:
            raise ValueError("OPTIONS_SINGLE_CSV must have 'symbol'.")
        df = df[df["symbol"] == tkr].copy()
    else:
        path = os.path.join(OPTIONS_DIR, f"{tkr}_options.csv")
        if not os.path.exists(path):
            print(f"[{tkr}] no options file")
            return pd.DataFrame()
        df = pd.read_csv(path)

    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    elif "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    else:
        raise ValueError(f"[{tkr}] no 'date'/'time'")

    if "expiry" not in df.columns and "expiration" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiration"]).dt.normalize()
    elif "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    else:
        raise ValueError(f"[{tkr}] no 'expiry'/'expiration'")

    if "strike" not in df.columns:
        raise ValueError(f"[{tkr}] no 'strike'")
    df["strike"] = df["strike"].astype(float)

    if "is_call" in df.columns:
        df["is_call"] = df["is_call"].astype(bool)
    else:
        right_col = next((c for c in ["right", "option_type", "cp_flag", "type"] if c in df.columns), None)
        if right_col is None:
            raise ValueError(f"[{tkr}] no call/put flag")
        df[right_col] = df[right_col].astype(str).str.upper().str.strip()
        df["is_call"] = df[right_col].isin(["C", "CALL"])

    cols = {c.lower(): c for c in df.columns}
    if "bid" in cols and "ask" in cols:
        df["mid"] = 0.5 * (df[cols["bid"]].astype(float) + df[cols["ask"]].astype(float))
    else:
        price_col = next((cols[c] for c in ["last", "lastprice", "close", "settle", "price"] if c in cols), None)
        if price_col is None:
            raise ValueError(f"[{tkr}] no price cols")
        df["mid"] = df[price_col].astype(float)

    df = df[df["mid"] > 0].copy()
    return df

def get_options_with_iv_local(
    tkr,
    close_long,
    start,
    end,
    r=R_RATE,
    q_div=Q_DIV,
    use_cache=USE_CACHE,
    cache_prefix=CACHE_PREFIX,
):
    cache_key = os.path.join(
        CACHE_DIR, f"{cache_prefix}_opt_{tkr}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    )

    # Try cache first
    if use_cache and os.path.exists(cache_key):
        try:
            df = pd.read_parquet(cache_key)
            if not df.empty and "iv" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
                return df
        except Exception as e:
            print(f"[{tkr}] cache read fail: {e}")

    # Load raw options
    df = _load_raw_options_for_ticker(tkr)
    if df.empty:
        print(f"[{tkr}] no raw options")
        return pd.DataFrame()

    # Time filter
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        print(f"[{tkr}] no options in range")
        return pd.DataFrame()

    # Attach spot
    if not isinstance(close_long.index, pd.MultiIndex):
        raise ValueError("close_long must be MultiIndex")

    key = pd.MultiIndex.from_arrays(
        [np.full(len(df), tkr), df["date"]],
        names=["symbol", "date"],
    )
    try:
        df["spot"] = close_long.reindex(key)["spot"].values
    except Exception:
        if tkr in close_long.columns:
            df["spot"] = close_long.reindex(df["date"])[tkr].values
        else:
            print(f"[{tkr}] no spot in close_long")
            return pd.DataFrame()

    df = df.dropna(subset=["spot"])
    if df.empty:
        print(f"[{tkr}] no rows after merge spot")
        return pd.DataFrame()

    # DTE
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    df = df[df["dte"] > 0].copy()
    if df.empty:
        print(f"[{tkr}] no positive DTE")
        return pd.DataFrame()

    # --- IV handling ---
    # If we have an implied_volatility column, keep it where it's usable and only compute IV
    # where it's missing / non-positive.
    if "implied_volatility" in df.columns:
        df["iv"] = _normalize_iv_scale(df["implied_volatility"])
    else:
        df["iv"] = np.nan

    mask_need_iv = df["iv"].isna() | (df["iv"] <= 0)

    if mask_need_iv.any():
        sub = df.loc[mask_need_iv].copy()

        def _compute_iv_row(rw):
            return _iv_from_price(
                float(rw["mid"]),
                float(rw["spot"]),
                float(rw["strike"]),
                max(1e-6, rw["dte"] / 365.0),
                bool(rw["is_call"]),
                r,
                q_div,
            )

        df.loc[mask_need_iv, "iv"] = [
            _compute_iv_row(rw) for _, rw in sub.iterrows()
        ]

    # Clean infinities / NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv"])
    if df.empty:
        print(f"[{tkr}] no valid IV")
        return pd.DataFrame()

    # --- Moneyness filter ---
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= MNY_MIN) & (df["moneyness"] <= MNY_MAX)].copy()
    if df.empty:
        print(f"[{tkr}] no options after moneyness filter {MNY_MIN:.2f}-{MNY_MAX:.2f}")
        return pd.DataFrame()

    df["symbol_under"] = tkr

    # Save a fully detailed cleaned version (original columns + added ones) back into alpha_option_raw
    if USE_ALPHA_OPTIONS:
        alpha_clean_path = os.path.join(ALPHA_OPTIONS_DIR, f"{tkr}_clean.parquet")
        try:
            df.to_parquet(alpha_clean_path, index=False)
            # print(f"[{tkr}] cleaned options saved -> {alpha_clean_path}")
        except Exception as e:
            print(f"[{tkr}] failed to save cleaned options: {e}")

    # Reduce to the columns the rest of the pipeline expects
    df_out = df[
        [
            "date",
            "expiry",
            "strike",
            "is_call",
            "spot",
            "mid",
            "iv",
            "dte",
            "moneyness",
            "symbol_under",
        ]
    ].sort_values(["date", "expiry", "strike"]).reset_index(drop=True)

    # Cache
    if use_cache:
        try:
            df_out.to_parquet(cache_key, index=False)
            print(f"[{tkr}] cached -> {cache_key}")
        except Exception as e:
            print(f"[{tkr}] cache save fail: {e}")

    return df_out


# ========= Build 3-case panels for one ticker =========
def build_atm_panels_for_ticker(
    opt_df, earn_tkr, pre, post,
    base_pre_mat_days, baseline_pre_td, baseline_post_td, max_anchor_lag_days
):
    """
    3 term cases × 3 moneyness tracks:
      MnyTrackID: 0=ATM, 1=+EV move, 2=−EV move
    """
    if opt_df.empty or earn_tkr.empty:
        return {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    df = opt_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.normalize()
    trading_days = pd.DatetimeIndex(sorted(df["date"].unique()))
    if len(trading_days) == 0:
        return {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame()}
    panels = {1: [], 2: [], 3: []}

    for _, ev in earn_tkr.iterrows():
        symbol = ev["symbol"]
        event_day = pd.to_datetime(ev["event_day"]).normalize()
        idx = trading_days.searchsorted(event_day)
        cand = [i for i in [idx, idx - 1, idx + 1] if 0 <= i < len(trading_days)]
        if not cand:
            continue
        anchor = min((trading_days[i] for i in cand), key=lambda d: abs((d - event_day).days))
        if abs((anchor - event_day).days) > max_anchor_lag_days:
            continue
        anchor_idx = int(np.where(trading_days == anchor)[0][0])

        # BMO realignment: shift anchor back by 1 trading day so that
        # t_rel=0 = last close BEFORE earnings are in the price.
        timing = str(ev.get("timing", "UNKNOWN")).upper()
        if timing == "BMO" and anchor_idx > 0:
            anchor_idx -= 1

        pre_idx = max(0, anchor_idx - pre)
        pre_date = trading_days[pre_idx]

        expiries_pre = sorted(df.loc[df["date"] == pre_date, "expiry"].unique())
        if not expiries_pre:
            continue
        future_exps = [e for e in expiries_pre if e > event_day]
        if not future_exps:
            continue
        exp1 = next((e for e in future_exps if e >= (event_day + BDay(6)).normalize()), None)
        exp2 = next((e for e in future_exps if (e - event_day).days >= base_pre_mat_days), None)
        exp3 = next((e for e in future_exps if (e - event_day).days >= 90), None)
        expiry_by_case = {1: exp1, 2: exp2, 3: exp3}
        if all(v is None for v in expiry_by_case.values()):
            continue

        b_left = max(0, anchor_idx - baseline_pre_td)
        b_right = max(0, anchor_idx - baseline_post_td)
        if b_right < b_left:
            b_left, b_right = b_right, b_left
        baseline_days = trading_days[b_left : b_right + 1]

        def _baseline_iv_for_expiry(target_expiry):
            ivs = []
            for dd in baseline_days:
                sub = df[(df["date"] == dd) & (df["expiry"] == target_expiry)]
                if sub.empty:
                    continue
                row = pick_atm_row(sub)
                if row is None or not np.isfinite(row["iv"]):
                    continue
                ivs.append(float(row["iv"]))
            return float(np.mean(ivs)) if ivs else np.nan

        for case in [1, 2, 3]:
            target = expiry_by_case[case]
            if target is None:
                continue
            baseline_iv = _baseline_iv_for_expiry(target)
            w_left = max(0, anchor_idx - pre)
            w_right = min(len(trading_days) - 1, anchor_idx + post)
            for i in range(w_left, w_right + 1):
                d = trading_days[i]
                sub = df[(df["date"] == d) & (df["expiry"] == target)]
                if sub.empty:
                    continue
                if "moneyness" not in sub.columns:
                    sub = sub.assign(moneyness=sub["strike"] / sub["spot"])
                row_atm = pick_atm_row(sub)
                if row_atm is None or not np.isfinite(row_atm["iv"]):
                    continue
                iv_atm = float(row_atm["iv"])
                spot_val = float(row_atm["spot"])
                EV_1d = iv_atm * np.sqrt(1.0 / 252.0)
                mny_up, mny_dn = float(np.exp(EV_1d)), float(np.exp(-EV_1d))

                def _pick(sub_df, target_mny):
                    tmp = sub_df.copy()
                    tmp["abs_mny_target"] = (tmp["moneyness"] - target_mny).abs()
                    return tmp.loc[tmp["abs_mny_target"].idxmin()]

                row_up = _pick(sub, mny_up)
                row_dn = _pick(sub, mny_dn)
                rel = i - anchor_idx
                rows_tracks = [(0, row_atm), (1, row_up), (2, row_dn)]

                for mny_id, row in rows_tracks:
                    iv_val = float(row["iv"])
                    panels[case].append(
                        dict(
                            symbol=symbol,
                            event_day=event_day,
                            case=case,
                            MnyTrackID=int(mny_id),
                            target_expiry=target,
                            date=d,
                            rel=rel,
                            iv=iv_val,
                            baseline_IV=baseline_iv,
                            abn_IV=iv_val - baseline_iv if np.isfinite(baseline_iv) else np.nan,
                            spot=float(row["spot"]),
                            strike=float(row["strike"]),
                            moneyness=float(row["moneyness"]),
                            is_call=bool(row["is_call"]),
                        )
                    )

    for c in [1, 2, 3]:
        panels[c] = (
            pd.DataFrame(panels[c])
            .sort_values(["symbol", "event_day", "case", "MnyTrackID", "date", "rel"])
            .reset_index(drop=True)
            if panels[c]
            else pd.DataFrame()
        )
    return panels

def build_opt_and_panels_for_ticker(tkr, earn_all, close_long,
                                    start, end, pre, post,
                                    baseline_pre_td, baseline_post_td,
                                    max_anchor_lag_days, base_pre_mat_days,
                                    keep_opt_data_in_ram, use_cache,
                                    cache_prefix, r, q_div):
    earn_tkr = earn_all[earn_all["symbol"]==tkr]
    if earn_tkr.empty:
        print(f"{tkr}: no earnings"); empty = {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}; return pd.DataFrame(),empty
    opt_df = get_options_with_iv_local(tkr,close_long,start,end,r,q_div,use_cache,cache_prefix)
    if opt_df.empty:
        print(f"{tkr}: no IV surface"); empty = {1:pd.DataFrame(),2:pd.DataFrame(),3:pd.DataFrame()}; return pd.DataFrame(),empty
    print(f"{tkr}: {len(opt_df):,} option rows")
    panels = build_atm_panels_for_ticker(opt_df,earn_tkr,pre,post,base_pre_mat_days,
                                         baseline_pre_td,baseline_post_td,max_anchor_lag_days)
    return (opt_df if keep_opt_data_in_ram else pd.DataFrame()), panels

# ========= Load spot & earnings =========
# ==== Load spot & earnings (BMO/AMC-aware) ====
close_long = load_spot_close_long(SPOT_CSV)
spot_min = close_long.index.get_level_values("date").min()
spot_max = close_long.index.get_level_values("date").max()
END_DATE = min(STUDY_END.normalize(), spot_max)
START_DATE = max(STUDY_START.normalize(), spot_min)
print(f"Study window: {START_DATE.date()} to {END_DATE.date()} ({spot_min.date()}–{spot_max.date()})")

earn = pd.read_csv(EARNINGS_CSV)
earn["symbol"] = earn["symbol"].astype(str).str.upper()
earn["event_day"] = pd.to_datetime(earn["event_day"]).dt.normalize()
earn["timing"] = earn.get("timing", "UNKNOWN")
earn["timing"] = earn["timing"].astype(str).str.upper()
earn = (
    earn[earn["symbol"].isin(UNIVERSE) &
         (earn["event_day"] >= START_DATE) &
         (earn["event_day"] <= END_DATE)]
    .drop_duplicates(subset=["symbol", "event_day"])
    .reset_index(drop=True)
)
print(f"Earnings events: {len(earn)}")

idx = close_long.index
mask = (
    idx.get_level_values("symbol").isin(UNIVERSE) &
    (idx.get_level_values("date") >= START_DATE) &
    (idx.get_level_values("date") <= END_DATE)
)
close_long = close_long[mask].sort_index()
print("close_long:", close_long.shape)

# ========= Build PANELS for all tickers =========
OPT_DATA, PANELS = {}, {}
for tkr in UNIVERSE:
    print(f"\nProcessing {tkr}...")
    opt_df_out, panels_tkr = build_opt_and_panels_for_ticker(
        tkr, earn, close_long, START_DATE, END_DATE,
        PRE, POST, BASELINE_PRE_TD, BASELINE_POST_TD,
        MAX_ANCHOR_LAG_DAYS, BASE_PRE_MAT_DAYS,
        KEEP_OPT_DATA_IN_RAM, USE_CACHE, CACHE_PREFIX, R_RATE, Q_DIV
    )
    OPT_DATA[tkr], PANELS[tkr] = opt_df_out, panels_tkr
print("\nAll tickers processed.")

# ==== panel_clean (now with MnyTrackID + timing) ====
clean_rows = []
for symbol, cases in PANELS.items():
    for case_id, df_case in cases.items():
        if df_case is None or df_case.empty:
            continue
        df_case["TTM_yrs"] = (df_case["target_expiry"] - df_case["date"]).dt.days / 365.0
        df = df_case.rename(
            columns={
                "symbol": "Symbol",
                "event_day": "EventDate",
                "case": "CaseID",
                "target_expiry": "MaturityDate",
                "date": "ObsDate",
                "rel": "t_rel",
                "iv": "IV",
                "baseline_IV": "IV_base",
                "abn_IV": "IV_abn",
            }
        )
        if "MnyTrackID" not in df.columns:
            df["MnyTrackID"] = 0
        clean_rows.append(
            df[
                [
                    "Symbol",
                    "EventDate",
                    "CaseID",
                    "MnyTrackID",
                    "MaturityDate",
                    "ObsDate",
                    "t_rel",
                    "TTM_yrs",
                    "IV",
                    "IV_base",
                    "IV_abn",
                ]
            ]
        )

panel_clean = (
    pd.concat(clean_rows, ignore_index=True)
    .sort_values(["Symbol", "EventDate", "CaseID", "MnyTrackID", "ObsDate", "t_rel"])
    .reset_index(drop=True)
    if clean_rows
    else pd.DataFrame(
        columns=[
            "Symbol",
            "EventDate",
            "CaseID",
            "MnyTrackID",
            "MaturityDate",
            "ObsDate",
            "t_rel",
            "TTM_yrs",
            "IV",
            "IV_base",
            "IV_abn",
        ]
    )
)
print("panel_clean:", panel_clean.shape)

earn_for_merge = earn[["symbol", "event_day", "timing"]].copy()
earn_for_merge["symbol"] = earn_for_merge["symbol"].astype(str).str.upper()
panel_clean = panel_clean.merge(
    earn_for_merge,
    left_on=["Symbol", "EventDate"],
    right_on=["symbol", "event_day"],
    how="left",
)
panel_clean["timing"] = panel_clean["timing"].fillna("UNKNOWN").str.upper()
panel_clean.drop(columns=["symbol", "event_day"], inplace=True)

# ==== ATM event_features (t_rel already realigned at panel build) ====
spot_df = pd.read_csv(SPOT_CSV, parse_dates=["date"])
spot_df["date"] = spot_df["date"].dt.normalize()
px = "adj_close" if "adj_close" in spot_df.columns else ("close" if "close" in spot_df.columns else None)
if px is None:
    raise ValueError("spot_data.csv must contain 'close' or 'adj_close'.")
spot_df = (
    spot_df[["symbol", "date", px]]
    .rename(columns={"symbol": "Symbol", "date": "Date", px: "Close"})
    .set_index(["Symbol", "Date"])
    .sort_index()
)

def _take_close(symbols, dates):
    idx = pd.MultiIndex.from_arrays([symbols, dates], names=["Symbol", "Date"])
    return spot_df.reindex(idx)["Close"].to_numpy()

anchors = (
    panel_clean[panel_clean["t_rel"] == 0]
    .groupby(["Symbol", "EventDate"])["ObsDate"]
    .min()
    .rename("AnchorDate")
    .reset_index()
)

pc1_atm = panel_clean[(panel_clean["CaseID"] == 1) & (panel_clean["MnyTrackID"] == 0)].copy()
pc3_atm = panel_clean[(panel_clean["CaseID"] == 3) & (panel_clean["MnyTrackID"] == 0)].copy()

def _pre_post(pc):
    # After BMO anchor realignment, t_rel=0 = pre and t_rel=1 = post for ALL events.
    pre = pc[pc["t_rel"] == 0].copy()
    post = pc[pc["t_rel"] == 1].copy()
    return pre, post

pre_c1_all, post_c1_all = _pre_post(pc1_atm)
pre_c1 = (
    pre_c1_all.groupby(["Symbol", "EventDate"], as_index=False)
    .agg(
        {
            "MaturityDate": "first",
            "IV": "mean",
            "IV_base": "mean",
            "IV_abn": "mean",
            "TTM_yrs": "mean",
        }
    )
    .rename(
        columns={
            "MaturityDate": "Maturity_pre",
            "IV": "IV_pre",
            "IV_base": "IV_base_pre",
            "IV_abn": "IV_abn_pre",
            "TTM_yrs": "TTM_pre_yrs",
        }
    )
)
post_c1 = (
    post_c1_all.groupby(["Symbol", "EventDate"], as_index=False)
    .agg({"IV": "mean", "IV_abn": "mean"})
    .rename(columns={"IV": "IV_post", "IV_abn": "IV_abn_post"})
)

pre_c3_all, _ = _pre_post(pc3_atm)
pre_c3 = (
    pre_c3_all.groupby(["Symbol", "EventDate"], as_index=False)
    .agg({"IV": "mean", "TTM_yrs": "mean"})
    .rename(columns={"IV": "IV_pre_long", "TTM_yrs": "TTM_pre_long_yrs"})
)

event_features = (
    anchors.merge(pre_c1, on=["Symbol", "EventDate"], how="left")
    .merge(post_c1, on=["Symbol", "EventDate"], how="left")
    .merge(pre_c3, on=["Symbol", "EventDate"], how="left")
)
event_features = event_features[~event_features["IV_pre"].isna()].copy()
event_features["Crush_1d"] = event_features["IV_post"] - event_features["IV_pre"]
event_features["TermSlope_pre"] = event_features["IV_pre"] - event_features["IV_pre_long"]
event_features["AnchorDate_plus1"] = event_features["AnchorDate"] + BDay(1)
event_features["S0"] = _take_close(event_features["Symbol"].values, event_features["AnchorDate"].values)
event_features["S1"] = _take_close(event_features["Symbol"].values, event_features["AnchorDate_plus1"].values)
event_features["R_0_1"] = np.log(event_features["S1"] / event_features["S0"])
event_features["EV_1d"] = event_features["IV_pre"] * np.sqrt(1 / 252.0)
event_features["VRP_1d"] = event_features["EV_1d"] - event_features["R_0_1"].abs()

# ========= Forward Variance =========
# The forward variance from T_short to T_long represents the market's implied
# variance for the period AFTER the short expiry — entirely post-earnings,
# no event risk. This is a cleaner non-event proxy than IV_long² alone.
#
# σ²_fwd = (IV_long² × T_long - IV_short² × T_short) / (T_long - T_short)

T_short = event_features["TTM_pre_yrs"]
T_long = event_features["TTM_pre_long_yrs"]
IV_short_sq = event_features["IV_pre"] ** 2
IV_long_sq = event_features["IV_pre_long"] ** 2

dt_fwd = T_long - T_short
# Guard against zero/negative forward period
dt_fwd_safe = dt_fwd.where(dt_fwd > 1e-6, np.nan)

event_features["Fwd_Var"] = (IV_long_sq * T_long - IV_short_sq * T_short) / dt_fwd_safe
event_features["Fwd_Vol"] = np.sqrt(event_features["Fwd_Var"].clip(lower=0))

# Spot-forward spread: how much extra variance the near term carries vs. forward
event_features["Spot_Fwd_Spread"] = IV_short_sq - event_features["Fwd_Var"]

# ========= Event Variance Decomposition =========
# Decompose total implied variance into regular (non-event) + event components.
# Uses forward variance as the non-event proxy (cleaner than IV_long²):
#
# Model: σ²_short × T_short = σ²_fwd × (T_short - 1/252) + σ²_event × (1/252)
#
# Solving for σ²_event (annualized):
#   σ²_event = (σ²_short × T_short - σ²_fwd × (T_short - 1/252)) × 252

event_features["Var_total_implied"] = IV_short_sq * T_short
event_features["Var_regular"] = event_features["Fwd_Var"]

event_features["Var_event_implied"] = (
    event_features["Var_total_implied"]
    - event_features["Var_regular"] * (T_short - 1.0 / 252.0)
) * 252.0

# Realized event variance (annualized from 1-day return)
event_features["Var_event_realized"] = event_features["R_0_1"] ** 2 * 252.0

# Implied and realized event moves (daily %, for intuition)
event_features["Implied_Event_Move"] = np.sqrt(
    event_features["Var_event_implied"].clip(lower=0) / 252.0
)
event_features["Realized_Event_Move"] = event_features["R_0_1"].abs()

# Event VRP: positive = market overpriced the event
event_features["Event_VRP"] = (
    event_features["Var_event_implied"] - event_features["Var_event_realized"]
)

# Normalized: fraction of implied event variance that was "excess"
event_features["Event_VRP_ratio"] = (
    event_features["Event_VRP"] / event_features["Var_event_implied"]
).replace([np.inf, -np.inf], np.nan)

event_features = event_features.replace([np.inf, -np.inf], np.nan)
event_features = event_features[
    [
        "Symbol",
        "EventDate",
        "AnchorDate",
        "Maturity_pre",
        "TTM_pre_yrs",
        "TTM_pre_long_yrs",
        "IV_base_pre",
        "IV_pre",
        "IV_abn_pre",
        "IV_post",
        "IV_abn_post",
        "Crush_1d",
        "IV_pre_long",
        "TermSlope_pre",
        "Fwd_Var",
        "Fwd_Vol",
        "Spot_Fwd_Spread",
        "S0",
        "AnchorDate_plus1",
        "S1",
        "R_0_1",
        "EV_1d",
        "VRP_1d",
        "Var_event_implied",
        "Var_event_realized",
        "Implied_Event_Move",
        "Realized_Event_Move",
        "Event_VRP",
        "Event_VRP_ratio",
    ]
].copy()
print("event_features:", event_features.shape)

# ==== 9-track event_features (3 cases × 3 mny tracks, explain only) ====
pc = panel_clean.copy()

def _make_pre_post_tracks(pc, case_id):
    sub = pc[pc["CaseID"] == case_id].copy()
    # After BMO anchor realignment, t_rel=0 = pre and t_rel=1 = post for ALL events.
    pre_all = sub[sub["t_rel"] == 0].copy()
    post_all = sub[sub["t_rel"] == 1].copy()
    pre = (
        pre_all.groupby(["Symbol", "EventDate", "MnyTrackID"], as_index=False)
        .agg(
            {
                "MaturityDate": "first",
                "IV": "mean",
                "IV_base": "mean",
                "IV_abn": "mean",
                "TTM_yrs": "mean",
            }
        )
        .rename(
            columns={
                "MaturityDate": "Maturity_pre",
                "IV": "IV_pre",
                "IV_base": "IV_base_pre",
                "IV_abn": "IV_abn_pre",
                "TTM_yrs": "TTM_pre_yrs",
            }
        )
    )
    post = (
        post_all.groupby(["Symbol", "EventDate", "MnyTrackID"], as_index=False)
        .agg({"IV": "mean", "IV_abn": "mean"})
        .rename(columns={"IV": "IV_post", "IV_abn": "IV_abn_post"})
    )
    pre["CaseID"] = case_id
    post["CaseID"] = case_id
    return pre.merge(post, on=["Symbol", "EventDate", "MnyTrackID", "CaseID"], how="left")

tracks_case1 = _make_pre_post_tracks(pc, 1)
tracks_case2 = _make_pre_post_tracks(pc, 2)
tracks_case3 = _make_pre_post_tracks(pc, 3)
event_features_all_tracks = pd.concat([tracks_case1, tracks_case2, tracks_case3], ignore_index=True)
event_features_all_tracks = event_features_all_tracks.merge(anchors, on=["Symbol", "EventDate"], how="left")
event_features_all_tracks["Crush_1d"] = event_features_all_tracks["IV_post"] - event_features_all_tracks["IV_pre"]
print("event_features_all_tracks:", event_features_all_tracks.shape)

# Computing the Aggregated panel for smart tables
# ==== Aggregate IV_abn over t_rel (ATM only) ====
agg = (
    panel_clean[panel_clean["MnyTrackID"] == 0]
    .groupby(["CaseID", "t_rel"], as_index=False)
    .agg(
        IV_abn_mean=("IV_abn", "mean"),
        TTM_yrs_mean=("TTM_yrs", "mean"),
    )
)
iv_pivot = (
    agg.pivot(index="t_rel", columns="CaseID", values="IV_abn_mean")
    .rename(columns={1: "IV_abn_Case1", 2: "IV_abn_Case2", 3: "IV_abn_Case3"})
)
dte_pivot = (
    (agg.pivot(index="t_rel", columns="CaseID", values="TTM_yrs_mean") * 365.0)
    .rename(columns={1: "DTE_days_Case1", 2: "DTE_days_Case2", 3: "DTE_days_Case3"})
)
df_agg = iv_pivot.join(dte_pivot, how="outer").sort_index().reset_index()

# ========= Per-symbol summary stats (vol behavior per name) =========
by_sym = (
    event_features.groupby("Symbol")
    .agg(
        N_events=("EventDate", "count"),
        Mean_IV_abn_pre=("IV_abn_pre", "mean"),
        Std_IV_abn_pre=("IV_abn_pre", "std"),
        Mean_TermSlope_pre=("TermSlope_pre", "mean"),
        Mean_Crush_1d=("Crush_1d", "mean"),
        Mean_VRP_1d=("VRP_1d", "mean"),
        Mean_Event_VRP=("Event_VRP", "mean"),
        Std_Event_VRP=("Event_VRP", "std"),
        Mean_Implied_Event_Move=("Implied_Event_Move", "mean"),
        Mean_Realized_Event_Move=("Realized_Event_Move", "mean"),
        Mean_Event_VRP_ratio=("Event_VRP_ratio", "mean"),
    )
    .reset_index()
)

# ========= Skew by side (upside vs downside) from tracks =========
# MnyTrackID convention from your build_atm_panels_for_ticker:
#   0 = ATM, 1 = +EV move (upside), 2 = -EV move (downside)
tracks = event_features_all_tracks.copy()

# ATM
atm = (
    tracks[tracks["MnyTrackID"] == 0]
    .rename(
        columns={
            "IV_pre": "IV_pre_ATM",
            "IV_abn_pre": "IV_abn_pre_ATM",
            "IV_post": "IV_post_ATM",
            "IV_abn_post": "IV_abn_post_ATM",
        }
    )
    [["Symbol", "EventDate", "CaseID", "Maturity_pre", "TTM_pre_yrs",
      "IV_pre_ATM", "IV_abn_pre_ATM", "IV_post_ATM", "IV_abn_post_ATM"]]
)

# Upside (+EV)
up = (
    tracks[tracks["MnyTrackID"] == 1]
    .rename(
        columns={
            "IV_pre": "IV_pre_UP",
            "IV_abn_pre": "IV_abn_pre_UP",
            "IV_post": "IV_post_UP",
            "IV_abn_post": "IV_abn_post_UP",
        }
    )
    [["Symbol", "EventDate", "CaseID",
      "IV_pre_UP", "IV_abn_pre_UP", "IV_post_UP", "IV_abn_post_UP"]]
)

# Downside (-EV)
down = (
    tracks[tracks["MnyTrackID"] == 2]
    .rename(
        columns={
            "IV_pre": "IV_pre_DOWN",
            "IV_abn_pre": "IV_abn_pre_DOWN",
            "IV_post": "IV_post_DOWN",
            "IV_abn_post": "IV_abn_post_DOWN",
        }
    )
    [["Symbol", "EventDate", "CaseID",
      "IV_pre_DOWN", "IV_abn_pre_DOWN", "IV_post_DOWN", "IV_abn_post_DOWN"]]
)

# Merge ATM / UP / DOWN per (Symbol, EventDate, CaseID)
skew_events = (
    atm.merge(up, on=["Symbol", "EventDate", "CaseID"], how="left")
       .merge(down, on=["Symbol", "EventDate", "CaseID"], how="left")
)

# Skew metrics (how high are wings vs ATM)
skew_events["Skew_IV_pre_UP"] = skew_events["IV_pre_UP"] - skew_events["IV_pre_ATM"]
skew_events["Skew_IV_pre_DOWN"] = skew_events["IV_pre_DOWN"] - skew_events["IV_pre_ATM"]

skew_events["Skew_IV_abn_pre_UP"] = (
    skew_events["IV_abn_pre_UP"] - skew_events["IV_abn_pre_ATM"]
)
skew_events["Skew_IV_abn_pre_DOWN"] = (
    skew_events["IV_abn_pre_DOWN"] - skew_events["IV_abn_pre_ATM"]
)

# Optional: skew on post IV (how much crush differs by wing)
skew_events["Skew_IV_post_UP"] = skew_events["IV_post_UP"] - skew_events["IV_post_ATM"]
skew_events["Skew_IV_post_DOWN"] = (
    skew_events["IV_post_DOWN"] - skew_events["IV_post_ATM"]
)

# Add anchor date to align with event_features
skew_events = skew_events.merge(
    anchors[["Symbol", "EventDate", "AnchorDate"]],
    on=["Symbol", "EventDate"],
    how="left",
)

print("skew_events:", skew_events.shape)

# Aggregate skew by symbol / case (so you see, e.g. AAPL has persistent downside skew)
skew_stats = (
    skew_events.groupby(["Symbol", "CaseID"])
    .agg(
        N_events=("EventDate", "nunique"),
        Mean_Skew_IV_pre_UP=("Skew_IV_pre_UP", "mean"),
        Mean_Skew_IV_pre_DOWN=("Skew_IV_pre_DOWN", "mean"),
        Mean_Skew_IV_abn_pre_UP=("Skew_IV_abn_pre_UP", "mean"),
        Mean_Skew_IV_abn_pre_DOWN=("Skew_IV_abn_pre_DOWN", "mean"),
    )
    .reset_index()
)
skew_case1 = skew_events[skew_events["CaseID"] == 1][[
    "Symbol",
    "EventDate",
    "Skew_IV_pre_UP",
    "Skew_IV_pre_DOWN",
    "Skew_IV_abn_pre_UP",
    "Skew_IV_abn_pre_DOWN",
]]
event_features = event_features.merge(
    skew_case1,
    on=["Symbol", "EventDate"],
    how="left",
)
print("skew_stats:", skew_stats.shape)

# ========= Plot data for Excel (unchanged logic) =========
iv_abn = event_features["IV_abn_pre"].dropna()
bins_iv = np.linspace(iv_abn.min(), iv_abn.max(), 51)
df_iv_hist = pd.DataFrame(
    {
        "BinLeft": bins_iv[:-1],
        "BinRight": bins_iv[1:],
        "Count": np.histogram(iv_abn, bins=bins_iv)[0],
    }
)
df_iv_vs_crush = event_features[["IV_abn_pre", "Crush_1d"]].dropna()
df_termslope_vs_crush = event_features[["TermSlope_pre", "Crush_1d"]].dropna()

vrp = event_features["VRP_1d"].dropna()
bins_vrp = np.linspace(vrp.min(), vrp.max(), 51)
df_vrp_hist = pd.DataFrame(
    {
        "BinLeft": bins_vrp[:-1],
        "BinRight": bins_vrp[1:],
        "Count": np.histogram(vrp, bins=bins_vrp)[0],
    }
)

# Event VRP plot data
evrp = event_features["Event_VRP"].dropna()
bins_evrp = np.linspace(evrp.quantile(0.01), evrp.quantile(0.99), 51)
df_evrp_hist = pd.DataFrame(
    {
        "BinLeft": bins_evrp[:-1],
        "BinRight": bins_evrp[1:],
        "Count": np.histogram(evrp.clip(bins_evrp[0], bins_evrp[-1]), bins=bins_evrp)[0],
    }
)
df_implied_vs_realized = event_features[["Implied_Event_Move", "Realized_Event_Move"]].dropna()
df_evrp_vs_ivabn = event_features[["IV_abn_pre", "Event_VRP"]].dropna()

# EventVRP persistence by symbol: autocorrelation lag-1 per ticker
_evrp_persist_rows = []
for sym, grp in event_features.dropna(subset=["Event_VRP"]).sort_values("EventDate").groupby("Symbol"):
    vals = grp["Event_VRP"].values
    n = len(vals)
    if n >= 4:
        ac1 = np.corrcoef(vals[:-1], vals[1:])[0, 1]
    else:
        ac1 = np.nan
    _evrp_persist_rows.append({
        "Symbol": sym,
        "N_events": n,
        "Mean_Event_VRP": np.mean(vals),
        "Std_Event_VRP": np.std(vals, ddof=1) if n > 1 else np.nan,
        "Pct_Positive": np.mean(vals > 0),
        "Autocorr_lag1": ac1,
        "Mean_Implied_Move": grp["Implied_Event_Move"].mean(),
        "Mean_Realized_Move": grp["Realized_Event_Move"].mean(),
        "Mean_Event_VRP_ratio": grp["Event_VRP_ratio"].mean(),
    })
df_evrp_by_symbol = pd.DataFrame(_evrp_persist_rows).sort_values("Mean_Event_VRP", ascending=False)

# ========= Single Excel export (all sheets) =========
with pd.ExcelWriter(OUT_EXCEL, engine="xlsxwriter") as w:
    # ================= README / Data dictionary =================
    readme_rows = [
        {
            "Section": "Overview",
            "Sheet": "EventFeatures_ATM",
            "Description": (
                "One row per (Symbol, EventDate) at-the-money (MnyTrackID=0, mainly Case 1 & 3). "
                "Pre-/post-earnings IV, crush, term slope, 1d return and variance risk premium."
            ),
        },
        {
            "Section": "Overview",
            "Sheet": "EventFeatures_Tracks",
            "Description": (
                "Pre-/post-earnings IV features for all moneyness tracks (MnyTrackID) and cases "
                "(1: short-dated, 2: mid, 3: long). Used to analyse skew and term structure around earnings."
            ),
        },
        {
            "Section": "Stats",
            "Sheet": "BySymbol_Stats",
            "Description": (
                "Per-symbol summary: number of earnings events, average abnormal IV, term-structure slope, "
                "average crush and variance risk premium."
            ),
        },
        {
            "Section": "Stats",
            "Sheet": "TermStructure_Agg",
            "Description": (
                "Average abnormal IV and corresponding average DTE (in days) across event-relative days (t_rel) "
                "and cases (1/2/3), ATM only."
            ),
        },
        {
            "Section": "Skew",
            "Sheet": "SkewBySide_Events",
            "Description": (
                "One row per (Symbol, EventDate, CaseID) with ATM vs upside (+EV) vs downside (-EV) IV levels "
                "and skew metrics (wing IV - ATM). This is the core table to plot skew by side."
            ),
        },
        {
            "Section": "Skew",
            "Sheet": "SkewBySide_Stats",
            "Description": (
                "Per-(Symbol, CaseID) averages of upside/downside skew, so you can see which names "
                "have persistent downside vs upside premia around earnings."
            ),
        },
        {
            "Section": "Plots",
            "Sheet": "Plot_IVabn_Hist",
            "Description": "Histogram of pre-earnings abnormal ATM IV (IV_abn_pre).",
        },
        {
            "Section": "Plots",
            "Sheet": "Plot_IVabn_vs_Crush",
            "Description": "Scatter of pre-earnings abnormal ATM IV vs 1d post-earnings IV crush.",
        },
        {
            "Section": "Plots",
            "Sheet": "Plot_TermSlope_vs_Crush",
            "Description": "Scatter of pre-earnings term-structure slope vs 1d IV crush.",
        },
        {
            "Section": "Plots",
            "Sheet": "Plot_VRP1d_Hist",
            "Description": "Histogram of 1d variance risk premium: EV_1d - |return|.",
        },
        {
            "Section": "EventVRP",
            "Sheet": "Plot_EventVRP_Hist",
            "Description": "Histogram of Event Variance Risk Premium (Event_VRP = implied event variance - realized event variance).",
        },
        {
            "Section": "EventVRP",
            "Sheet": "Plot_Implied_vs_Realized",
            "Description": "Scatter of implied event move vs realized event move. Points below 45-degree line = market overpriced.",
        },
        {
            "Section": "EventVRP",
            "Sheet": "Plot_EventVRP_vs_IVabn",
            "Description": "Scatter of pre-earnings abnormal IV vs Event VRP: does higher vol buildup predict larger overpricing?",
        },
        {
            "Section": "EventVRP",
            "Sheet": "EventVRP_BySymbol",
            "Description": (
                "Per-symbol Event VRP summary: mean, std, % positive, lag-1 autocorrelation (persistence). "
                "Use this to identify names with persistent overpricing of earnings events."
            ),
        },
        {
            "Section": "Regression",
            "Sheet": "Reg_EventVRP_Data",
            "Description": (
                "Cross-sectional dataset used to regress Event_VRP on pre-earnings IV_abn, "
                "term slope, skew and underlying return."
            ),
        },
        {
            "Section": "Regression",
            "Sheet": "Reg_EventVRP_Coef",
            "Description": "OLS coefficients for Event_VRP ~ IV_abn_pre_z + TermSlope_pre_z + Skew_z + R_0_1.",
        },
        {
            "Section": "Regression",
            "Sheet": "Reg_Crush_1d_Data",
            "Description": (
                "Cross-sectional dataset used to regress 1d IV crush on pre-earnings IV_abn, "
                "term slope, VRP and underlying return."
            ),
        },
        {
            "Section": "Regression",
            "Sheet": "Reg_Crush_1d_Coef",
            "Description": "OLS coefficients for Crush_1d ~ IV_abn_pre_z + TermSlope_pre_z + VRP_1d_z + R_0_1.",
        },
        {
            "Section": "Regression",
            "Sheet": "EventVRP_Quintiles",
            "Description": (
                "Average Event_VRP by composite signal quintile, where the composite signal is a "
                "blend of vol buildup and term-structure slope. Also shows PnL_proxy for comparison."
            ),
        },
        {
            "Section": "Panels",
            "Sheet": "PANELS_*",
            "Description": (
                "Per-ticker raw IV panels for each case. One sheet per (symbol, case): "
                "full daily IV path, baseline IV and abnormal IV around earnings."
            ),
        },
    ]
    pd.DataFrame(readme_rows).to_excel(w, sheet_name="README", index=False)

    # ================= Core feature tables =================
    event_features.to_excel(w, sheet_name="EventFeatures_ATM", index=False)
    event_features_all_tracks.to_excel(w, sheet_name="EventFeatures_Tracks", index=False)
    by_sym.to_excel(w, sheet_name="BySymbol_Stats", index=False)
    df_agg.to_excel(w, sheet_name="TermStructure_Agg", index=False)

    # ================= Skew by side tables =================
    skew_events.to_excel(w, sheet_name="SkewBySide_Events", index=False)
    skew_stats.to_excel(w, sheet_name="SkewBySide_Stats", index=False)

    # ================= Plot data =================
    df_iv_hist.to_excel(w, sheet_name="Plot_IVabn_Hist", index=False)
    df_iv_vs_crush.to_excel(w, sheet_name="Plot_IVabn_vs_Crush", index=False)
    df_termslope_vs_crush.to_excel(w, sheet_name="Plot_TermSlope_vs_Crush", index=False)
    df_vrp_hist.to_excel(w, sheet_name="Plot_VRP1d_Hist", index=False)

    # ================= Event VRP sheets =================
    df_evrp_hist.to_excel(w, sheet_name="Plot_EventVRP_Hist", index=False)
    df_implied_vs_realized.to_excel(w, sheet_name="Plot_Implied_vs_Realized", index=False)
    df_evrp_vs_ivabn.to_excel(w, sheet_name="Plot_EventVRP_vs_IVabn", index=False)
    df_evrp_by_symbol.to_excel(w, sheet_name="EventVRP_BySymbol", index=False)

    # ================= Event VRP Regression =================
    df_evrp_reg = event_features.dropna(
        subset=["Event_VRP", "IV_abn_pre", "TermSlope_pre", "Spot_Fwd_Spread", "R_0_1"]
    ).copy()

    if not df_evrp_reg.empty:
        def _zscore(series: pd.Series) -> pd.Series:
            std = series.std()
            if not np.isfinite(std) or std == 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - series.mean()) / std

        for c in ["IV_abn_pre", "TermSlope_pre", "Spot_Fwd_Spread"]:
            df_evrp_reg[c + "_z"] = _zscore(df_evrp_reg[c])

        # Add skew if available
        skew_col = None
        if "Skew_IV_abn_pre_DOWN" in df_evrp_reg.columns:
            df_evrp_reg["Skew_z"] = _zscore(df_evrp_reg["Skew_IV_abn_pre_DOWN"].fillna(0))
            skew_col = "Skew_z"

        x_cols = ["IV_abn_pre_z", "TermSlope_pre_z", "Spot_Fwd_Spread_z", "R_0_1"]
        if skew_col:
            x_cols.append(skew_col)

        y_evrp = df_evrp_reg["Event_VRP"]
        X_evrp = sm.add_constant(df_evrp_reg[x_cols])
        model_evrp = sm.OLS(y_evrp, X_evrp).fit(cov_type="HC3")
        print("\n========= Event VRP Regression =========")
        print(model_evrp.summary())

        df_evrp_reg["Event_VRP_hat"] = model_evrp.predict(X_evrp)

        coef_evrp = (
            model_evrp.summary2()
            .tables[1]
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
        # Append model summary stats as extra rows
        stats_evrp = pd.DataFrame([
            {"Variable": "R_squared", "Coef.": model_evrp.rsquared},
            {"Variable": "R_squared_adj", "Coef.": model_evrp.rsquared_adj},
            {"Variable": "F_statistic", "Coef.": model_evrp.fvalue},
            {"Variable": "F_pvalue", "Coef.": model_evrp.f_pvalue},
            {"Variable": "N_obs", "Coef.": model_evrp.nobs},
        ])
        coef_evrp = pd.concat([coef_evrp, stats_evrp], ignore_index=True)

        df_evrp_reg.to_excel(w, sheet_name="Reg_EventVRP_Data", index=False)
        coef_evrp.to_excel(w, sheet_name="Reg_EventVRP_Coef", index=False)
    else:
        pd.DataFrame().to_excel(w, sheet_name="Reg_EventVRP_Data", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Reg_EventVRP_Coef", index=False)

    # ================= Regression / signal summary =================
    df_reg = event_features.dropna(
        subset=["Crush_1d", "IV_abn_pre", "TermSlope_pre", "VRP_1d", "R_0_1"]
    ).copy()

    if not df_reg.empty:
        def _zscore(series: pd.Series) -> pd.Series:
            std = series.std()
            if not np.isfinite(std) or std == 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - series.mean()) / std

        for c in ["IV_abn_pre", "TermSlope_pre", "VRP_1d"]:
            df_reg[c + "_z"] = _zscore(df_reg[c])

        y = df_reg["Crush_1d"]
        X = sm.add_constant(df_reg[["IV_abn_pre_z", "TermSlope_pre_z", "VRP_1d_z", "R_0_1"]])
        model = sm.OLS(y, X).fit(cov_type="HC3")
        print(model.summary())

        # Reconstructed crush from regression betas (predicted values)
        df_reg["Crush_1d_hat"] = model.predict(X)

        coef_table = (
            model.summary2()
            .tables[1]
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
        # Append model summary stats as extra rows
        stats_crush = pd.DataFrame([
            {"Variable": "R_squared", "Coef.": model.rsquared},
            {"Variable": "R_squared_adj", "Coef.": model.rsquared_adj},
            {"Variable": "F_statistic", "Coef.": model.fvalue},
            {"Variable": "F_pvalue", "Coef.": model.f_pvalue},
            {"Variable": "N_obs", "Coef.": model.nobs},
        ])
        coef_table = pd.concat([coef_table, stats_crush], ignore_index=True)

        df_reg["Signal_VolBuildup"] = -df_reg["IV_abn_pre_z"]
        df_reg["Signal_TermSlope"] = -df_reg["TermSlope_pre_z"]
        df_reg["Signal_Composite"] = (
            0.6 * df_reg["Signal_VolBuildup"] + 0.4 * df_reg["Signal_TermSlope"]
        )

        df_reg["ImpliedMove_1d"] = df_reg["EV_1d"]
        df_reg["RealizedMove_1d"] = df_reg["R_0_1"].abs()
        df_reg["IVCrush_Impact"] = -df_reg["Crush_1d"] * np.sqrt(1 / 252)
        df_reg["PnL_proxy"] = (
            df_reg["IVCrush_Impact"] + df_reg["RealizedMove_1d"] - df_reg["ImpliedMove_1d"]
        )

        # Also bring Event_VRP into df_reg for quintile analysis
        if "Event_VRP" in event_features.columns:
            _evrp_merge = event_features[["Symbol", "EventDate", "Event_VRP", "Implied_Event_Move", "Realized_Event_Move"]].copy()
            df_reg = df_reg.merge(_evrp_merge, on=["Symbol", "EventDate"], how="left", suffixes=("", "_dup"))
            # Drop any duplicate columns from merge
            df_reg = df_reg[[c for c in df_reg.columns if not c.endswith("_dup")]]

        df_reg["Signal_quintile"] = pd.qcut(
            df_reg["Signal_Composite"], 5, labels=False, duplicates="drop"
        ) + 1

        # Quintile performance: both PnL_proxy (legacy) and Event_VRP (new)
        agg_cols = {"PnL_proxy": "mean"}
        rename_map = {"PnL_proxy": "MeanPnL_proxy"}
        if "Event_VRP" in df_reg.columns:
            agg_cols["Event_VRP"] = "mean"
            rename_map["Event_VRP"] = "MeanEventVRP"
        perf = (
            df_reg.groupby("Signal_quintile")
            .agg(agg_cols)
            .rename(columns=rename_map)
        )

        df_reg.to_excel(w, sheet_name="Reg_Crush_1d_Data", index=False)
        perf.to_excel(w, sheet_name="EventVRP_Quintiles")
        coef_table.to_excel(w, sheet_name="Reg_Crush_1d_Coef", index=False)
    else:
        pd.DataFrame().to_excel(w, sheet_name="Reg_Crush_1d_Data", index=False)
        pd.DataFrame().to_excel(w, sheet_name="EventVRP_Quintiles", index=False)
        pd.DataFrame().to_excel(w, sheet_name="Reg_Crush_1d_Coef", index=False)

    # ================= Ticker-level panels =================
    for tkr, cases in PANELS.items():
        for case_id, df_case in cases.items():
            if df_case is None or df_case.empty:
                continue
            sheet = f"PANELS_{tkr}_c{case_id}"[:31]
            cols = [
                "symbol",
                "event_day",
                "case",
                "MnyTrackID",
                "target_expiry",
                "date",
                "rel",
                "iv",
                "baseline_IV",
                "abn_IV",
                "spot",
                "strike",
                "moneyness",
                "is_call",
            ]
            df_case[cols].to_excel(w, sheet_name=sheet, index=False)

print(f"All outputs written to {OUT_EXCEL}")
