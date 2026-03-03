import io
import time
import pathlib
import requests
import pandas as pd
import datetime as dt
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ALPHA_KEY')
BASE_URL = "https://www.alphavantage.co/query"

# ------------------------------------------
# CONFIG – user-facing
# ------------------------------------------
# Date range you want in final dataframe
USER_START_DATE = "2020-01-01"
USER_END_DATE   = "2025-11-16"   # or pd.Timestamp.today().strftime("%Y-%m-%d")

# Intraday interval: "1min", "5min", "15min", "30min", "60min"
INTERVAL = "15min"

SYMBOLS = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT","QQQ","SPY"
]
#
# SYMBOLS = [
# "ACN","BKNG","GEV",
#     "SPGI","ANET","KLAC","BSX","PFE","SYK","WELL","UNP","PGR","COF",
#     "DE","LOW","MDT","ETN","PANW","CRWD","HON","CB","PLD","ADI"
#
#     "HCA","BX","VRTX","COP","MCK","LMT","PH","KKR","CEG","ADP",
#     "CMCSA","CVS","CME","SO","SBUX","HOOD","DUK","BMY","GD","NEM",
#     "TT","MMM","MMC","ICE","WM","MCO","ORLY","AMT","SHW","DELL",
#     "CDNS","DASH","NOC","REGN","HWM","MAR","TDG","ECL","APO","CTAS",
#     "AON","CI","USB","BK","EQIX","MDLZ","PNC","WMB","SNPS","EMR"
#
#     "RCL","ITW","ELV","COR","MNST","JCI","ABNB","GLW","RSG","CL",
#     "CMI","AZO","COIN","TRV","AJG","AEP","TEL","NSC","PWR","CSX",
#     "HLT","FDX","ADSK","MSI","SRE","WDAY","KMI","SPG","FTNT","TFC",
#     "AFL","EOG","IDXX","WBD","MPC","APD","FCX","VST","ROST","ALL",
#     "DDOG","BDX","PCAR","SLB","DLR","PSX","ZTS","VLO","D","O"
#
#     "D","O","F","LHX","NDAQ","URI","EA","MET","CAH","EW",
#     "BKR","NXPI","XEL","PSA","CBRE","ROP","EXC","DHI","FAST","CARR",
#     "LVS","OKE","CMG","CTVA","AME","TTWO","GWW","KR","MPWR","ROK",
#     "FANG","ETR","A","FICO","AXON","YUM","MSCI","AMP","DAL","PEG",
#     "OXY","AIG","TGT","XYZ","PAYX","CCI","VMC","IQV","HIG","HSY"
#
#     "EQT","KDP","PRU","VTR","CPRT","TRGP","MLM","GRMN","CTSH","EBAY",
#     "RMD","NUE","SYY","WEC","GEHC","ED","KMB","WAB","PCG","OTIS",
#     "XYL","FIS","ACGL","KEYS","EL","CCL","STT","KVUE","FISV","UAL",
#     "NRG","LEN","IR","VRSK","EXPE","VICI","RJF","WTW","LYV","KHC",
#     "MTD","ROL","HUM","WRB","FSLR","MTB","ADM","CSGP","K","EXE"
#
#     "IBKR","MCHP","TSCO","HPE","AEE","FITB","ATO","TER","DTE","ODFL",
#     "EXR","SYF","FE","EME","BRO","PPL","CBOE","BIIB","BR","CINF",
#     "STE","CNP","EFX","CHTR","AVB","IRM","HBAN","DOV","AWK","GIS",
#     "ES","VLTO","NTRS","PHM","DXCM","STLD","LDOS","ULTA","DG","WAT",
#     "STZ","EQR","TDY","VRSN","DVN","CFG","PODD","CMS","HUBB","HPQ"
# ]  # 300-350

OUT_DIR = pathlib.Path("alpha_intraday_15m_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Throttle: adapt to your plan; for 75 req/min -> ~0.9–1.0s
REQUEST_SLEEP_SECONDS = 0.9


# ------------------------------------------
# Helpers – month range
# ------------------------------------------
def month_range(start_date: str, end_date: str):
    """
    Generate YYYY-MM strings for all calendar months overlapping
    [start_date, end_date], inclusive.
    """
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)

    # Normalize to first day of month
    cur = dt.date(start_ts.year, start_ts.month, 1)
    last = dt.date(end_ts.year, end_ts.month, 1)

    months = []
    while cur <= last:
        months.append(f"{cur.year:04d}-{cur.month:02d}")
        # Increment month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return months


# ------------------------------------------
# Low-level API call: intraday by month
# ------------------------------------------
def fetch_intraday_month(symbol: str,
                         interval: str,
                         month_str: str) -> pd.DataFrame:
    """
    Fetch intraday data for a given symbol, interval, and month (YYYY-MM).
    Uses TIME_SERIES_INTRADAY with month=YYYY-MM & outputsize=full.
    Returns empty df on 'no data' or parsing failure.
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "month": month_str,          # 'YYYY-MM'
        "outputsize": "full",
        "datatype": "csv",
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    text = resp.text

    # Alpha Vantage may return JSON error/Note even if datatype=csv
    if text.startswith("{") or "Error Message" in text or "Thank you for using" in text:
        print(f"[WARN] {symbol} {interval} {month_str} returned non-CSV message (likely limit or error).")
        print(f"       First 120 chars: {text[:120]!r}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"[ERROR] Failed to parse CSV for {symbol} {interval} {month_str}: {e}")
        print(f"        First 120 chars: {text[:120]!r}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Expected columns: time, open, high, low, close, volume
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])  # AV timestamps are NY time (naive)

    df["symbol"] = symbol
    df["interval"] = interval
    df["month"] = month_str

    return df


# ------------------------------------------
# Main per-symbol downloader
# ------------------------------------------
def download_intraday_symbol(symbol: str,
                             interval: str,
                             user_start_date: str,
                             user_end_date: str,
                             sleep_seconds: float = REQUEST_SLEEP_SECONDS):
    print(f"\n======================")
    print(f"START {symbol} intraday {interval}")
    print(f"Range: {user_start_date} -> {user_end_date}")
    print(f"======================")

    start_ts = pd.to_datetime(user_start_date)
    end_ts   = pd.to_datetime(user_end_date)

    months = month_range(user_start_date, user_end_date)
    print(f"[INFO] {symbol}: will query {len(months)} month(s): {months}")

    all_chunks = []
    failed = []

    t0 = time.time()
    total_requests = 0
    total_rows = 0

    for idx, month_str in enumerate(months, start=1):
        api_symbol = symbol

        try:
            df = fetch_intraday_month(api_symbol, interval, month_str)
        except Exception as e:
            print(f"[ERROR] HTTP error for {symbol} ({api_symbol}) {interval} {month_str}: {e}")
            failed.append({
                "symbol": symbol,
                "api_symbol": api_symbol,
                "interval": interval,
                "month": month_str,
                "error": str(e)
            })
            time.sleep(sleep_seconds)
            continue

        total_requests += 1

        if df.empty:
            failed.append({
                "symbol": symbol,
                "api_symbol": api_symbol,
                "interval": interval,
                "month": month_str,
                "error": "empty_or_error"
            })
        else:
            # Filter exact date range
            if "time" in df.columns:
                df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]

            if not df.empty:
                all_chunks.append(df)
                total_rows += len(df)

        # Progress log
        elapsed = time.time() - t0
        print(
            f"[PROGRESS] {symbol}: month {idx}/{len(months)} ({month_str}) | "
            f"requests={total_requests} | rows_so_far={total_rows} | "
            f"elapsed={elapsed / 60:.1f} min"
        )
        if not df.empty:
            print(f"           Last month {month_str}: df.shape = {df.shape}")

        time.sleep(sleep_seconds)

    # Concatenate and save to parquet
    if all_chunks:
        full_df = pd.concat(all_chunks, axis=0, ignore_index=True)

        # Enforce numeric types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

        # Sort by time
        if "timestamp" in full_df.columns:
            full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        out_file = OUT_DIR / f"{symbol}_intraday_{interval}_{user_start_date}_to_{user_end_date}.parquet"
        full_df.to_parquet(out_file, index=False)
        print(f"[DONE] {symbol}: saved {len(full_df):,} rows to {out_file}")
    else:
        print(f"[WARN] {symbol}: no successful data at all in the requested range, nothing saved.")

    # Save failed months for re-run
    if failed:
        fail_df = pd.DataFrame(failed)
        fail_file = OUT_DIR / f"{symbol}_intraday_failed_{interval}_{user_start_date}_to_{user_end_date}.csv"
        fail_df.to_csv(fail_file, index=False)
        print(f"[INFO] {symbol}: {len(failed)} failed/empty months logged in {fail_file}")
    else:
        print(f"[INFO] {symbol}: 0 failed months.")

    elapsed_total = time.time() - t0
    print(
        f"[SUMMARY] {symbol}: requests={total_requests}, total_rows={total_rows:,}, "
        f"elapsed={elapsed_total / 60:.1f} minutes"
    )


# ------------------------------------------
# Run for all symbols
# ------------------------------------------
if __name__ == "__main__":
    for sym in SYMBOLS:
        download_intraday_symbol(
            sym,
            INTERVAL,
            USER_START_DATE,
            USER_END_DATE,
            sleep_seconds=REQUEST_SLEEP_SECONDS,
        )
