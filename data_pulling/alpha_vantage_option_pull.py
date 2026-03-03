import io
import time
import pathlib
import requests
import pandas as pd
import datetime
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('ALPHA_KEY')
BASE_URL = "https://www.alphavantage.co/query"

# ------------------------------------------
# CONFIG
# ------------------------------------------
START_DATE = "2020-01-01"
END_DATE   = "2025-11-16"   # or pd.Timestamp.today().strftime("%Y-%m-%d")
SYMBOLS_top50 = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK.B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT","QQQ","SPY"
]

# SYMBOLS = [
#     "GE","MS","AXP","PM","TMUS","MU","GS","RTX","ABT","TMO",
#     "ISRG","INTU","AMGN","LRCX","C","APP","QCOM","UBER","NEE","VZ",
#     "NOW","TJX","APH","SCHW","DHR","BLK","GILD","ACN","BKNG","GEV",
#     "SPGI","ANET","KLAC","BSX","PFE","SYK","WELL","UNP","PGR","COF",
#     "DE","LOW","MDT","ETN","PANW","CRWD","HON","CB","PLD","ADI"
# ]
#
# SYMBOLS = [
#     "HCA","BX","VRTX","COP","MCK","LMT","PH","KKR","CEG","ADP",
#     "CMCSA","CVS","CME","SO","SBUX","HOOD","DUK","BMY","GD","NEM",
#     "TT","MMM","MMC","ICE","WM","MCO","ORLY","AMT","SHW","DELL",
#     "CDNS","DASH","NOC","REGN","HWM","MAR","TDG","ECL","APO","CTAS",
#     "AON","CI","USB","BK","EQIX","MDLZ","PNC","WMB","SNPS","EMR"
# ]
#
# SYMBOLS = [
#     "RCL","ITW","ELV","COR","MNST","JCI","ABNB","GLW","RSG","CL",
#     "CMI","AZO","COIN","TRV","AJG","AEP","TEL","NSC","PWR","CSX",
#     "HLT","FDX","ADSK","MSI","SRE","WDAY","KMI","SPG","FTNT","TFC",
#     "AFL","EOG","IDXX","WBD","MPC","APD","FCX","VST","ROST","ALL",
#     "DDOG","BDX","PCAR","SLB","DLR","PSX","ZTS","VLO","D","O"
# ]

# SYMBOLS = [
#     "D","O","F","LHX","NDAQ","URI","EA","MET","CAH","EW",
#     "BKR","NXPI","XEL","PSA","CBRE","ROP","EXC","DHI","FAST","CARR",
#     "LVS","OKE","CMG","CTVA","AME","TTWO","GWW","KR","MPWR","ROK",
#     "FANG","ETR","A","FICO","AXON","YUM","MSCI","AMP","DAL","PEG",
#     "OXY","AIG","TGT","XYZ","PAYX","CCI","VMC","IQV","HIG","HSY"
# ]

# SYMBOLS = [
#     "EQT","KDP","PRU","VTR","CPRT","TRGP","MLM","GRMN","CTSH","EBAY",
#     "RMD","NUE","SYY","WEC","GEHC","ED","KMB","WAB","PCG","OTIS",
#     "XYL","FIS","ACGL","KEYS","EL","CCL","STT","KVUE","FISV","UAL",
#     "NRG","LEN","IR","VRSK","EXPE","VICI","RJF","WTW","LYV","KHC",
#     "MTD","ROL","HUM","WRB","FSLR","MTB","ADM","CSGP","K","EXE"
# ]

SYMBOLS = [
    "IBKR","MCHP","TSCO","HPE","AEE","FITB","ATO","TER","DTE","ODFL",
    "EXR","SYF","FE","EME","BRO","PPL","CBOE","BIIB","BR","CINF",
    "STE","CNP","EFX","CHTR","AVB","IRM","HBAN","DOV","AWK","GIS",
    "ES","VLTO","NTRS","PHM","DXCM","STLD","LDOS","ULTA","DG","WAT",
    "STZ","EQR","TDY","VRSN","DVN","CFG","PODD","CMS","HUBB","HPQ"
]  # 300-350

OUT_DIR = pathlib.Path("alpha_options_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Throttle: 75 requests/min -> 60/75 ≈ 0.8s, use 0.9–1.0s to be safe
REQUEST_SLEEP_SECONDS = 0.9


# ------------------------------------------
# Low-level API call
# ------------------------------------------
def fetch_option_chain(symbol: str, date: str) -> pd.DataFrame:
    """Fetch full option chain for symbol on given date.
    Raises on HTTP error. Returns empty df on 'no data' or parsing failure."""
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "date": date,   # 'YYYY-MM-DD'
        "datatype": "csv",
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    text = resp.text

    # Alpha Vantage sometimes returns JSON error/Note even if datatype=csv
    if text.startswith("{") or "Error Message" in text or "Thank you for using" in text:
        # This is usually rate limit or invalid call
        print(f"[WARN] {symbol} {date} returned non-CSV message (likely limit or error).")
        print(f"       First 120 chars: {text[:120]!r}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        print(f"[ERROR] Failed to parse CSV for {symbol} {date}: {e}")
        print(f"        First 120 chars: {resp[:120]!r}")
        return pd.DataFrame()

    # Standardise columns with explicit symbol/date in case API changes
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])

    return df


# ------------------------------------------
# Utility: generate dates
# ------------------------------------------
def generate_trading_like_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Approximate trading days using business days; good enough for your purpose.
       If you want exact exchange holidays, swap to pandas_market_calendars later."""
    days = pd.bdate_range(start=start_date, end=end_date, freq="B")
    return days


# ------------------------------------------
# Main per-symbol downloader
# ------------------------------------------
def download_symbol(symbol: str,
                    start_date: str,
                    end_date: str,
                    sleep_seconds: float = REQUEST_SLEEP_SECONDS):
    print(f"\n======================")
    print(f"START {symbol} from {start_date} to {end_date}")
    print(f"======================")

    dates = generate_trading_like_dates(start_date, end_date)
    print(f"[INFO] {symbol}: {len(dates)} business days to query")

    all_chunks = []
    failed = []

    t0 = time.time()
    total_requests = 0
    total_rows = 0

    for idx, dt in enumerate(dates, start=1):
        date_str = dt.strftime("%Y-%m-%d")
        date_obj = dt.date()

        # === Minimal META fix ===
        if symbol == "META" and date_obj < datetime.date(2022, 6, 9):
            api_symbol = "FB"  # query AlphaVantage correctly
        else:
            api_symbol = symbol  # use original ticker

        try:
            df = fetch_option_chain(api_symbol, date_str)
        except Exception as e:
            print(f"[ERROR] HTTP error for {symbol} ({api_symbol}) {date_str}: {e}")
            failed.append({"symbol": symbol, "api_symbol": api_symbol, "date": date_str, "error": str(e)})
            time.sleep(sleep_seconds)
            continue

        total_requests += 1

        if df.empty:
            failed.append({"symbol": symbol, "api_symbol": api_symbol, "date": date_str, "error": "empty_or_error"})
        else:
            # regardless of FB/META used for request, store as META
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["date"])
            all_chunks.append(df)
            total_rows += len(df)

        # Progress log (unchanged)
        if idx % 10 == 0 or idx == 1:
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] {symbol}: day {idx}/{len(dates)} | "
                f"requests={total_requests} | rows_so_far={total_rows} | "
                f"elapsed={elapsed / 60:.1f} min"
            )
            if not df.empty:
                print(f"           Last date {date_str}: df.shape = {df.shape}")

        time.sleep(sleep_seconds)

    # Concatenate and save
    if all_chunks:
        full_df = pd.concat(all_chunks, axis=0, ignore_index=True)
        # Force numeric types for columns that should be floats/ints
        numeric_cols = [
            "strike",
            "implied_volatility",
            "open_interest",
            "volume",
            "bid",
            "ask",
            "last",
            # add others from the CSV if present
        ]

        for col in numeric_cols:
            if col in full_df.columns:
                full_df[col] = pd.to_numeric(full_df[col], errors="coerce")

        out_file = OUT_DIR / f"{symbol}.parquet"
        # out_file = OUT_DIR / f"{symbol}_options_{start_date}_to_{end_date}.parquet"
        full_df.to_parquet(out_file, index=False)
        print(f"[DONE] {symbol}: saved {len(full_df):,} rows to {out_file}")
    else:
        print(f"[WARN] {symbol}: no successful data at all, nothing saved.")

    # Save failed dates for re-run
    if failed:
        fail_df = pd.DataFrame(failed)
        fail_file = OUT_DIR / f"{symbol}_failed_{start_date}_to_{end_date}.csv"
        fail_df.to_csv(fail_file, index=False)
        print(f"[INFO] {symbol}: {len(failed)} failed/empty dates logged in {fail_file}")
    else:
        print(f"[INFO] {symbol}: 0 failed dates.")

    elapsed_total = time.time() - t0
    print(
        f"[SUMMARY] {symbol}: requests={total_requests}, total_rows={total_rows:,}, "
        f"elapsed={elapsed_total/60:.1f} minutes"
    )


# ------------------------------------------
# Run for all symbols
# ------------------------------------------
if __name__ == "__main__":
    for sym in SYMBOLS:
        download_symbol(sym, START_DATE, END_DATE)