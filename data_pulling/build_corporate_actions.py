import time
import json
import pathlib
import requests
import pandas as pd

# ================== CONFIG ==================
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ALPHA_KEY')
BASE_URL = "https://www.alphavantage.co/query"

SYMBOLS = [
    "NVDA","MSFT","AAPL","AMZN","META","AVGO","GOOGL","GOOG","BRK-B","TSLA","JPM","V","LLY","NFLX","XOM","MA","WMT",
    "COST","ORCL","JNJ","HD","PG","ABBV","BAC","UNH","CRM","ADBE","PYPL","AMD","INTC","CSCO","MCD","NKE","WFC","CVX",
    "PEP","KO","DIS","BA","MRK","MO","IBM","T","GM","CAT","UPS","DOW","PLTR","TXN","LIN","AMAT",
"ACN","BKNG","GEV",
    "SPGI","ANET","KLAC","BSX","PFE","SYK","WELL","UNP","PGR","COF",
    "DE","LOW","MDT","ETN","PANW","CRWD","HON","CB","PLD","ADI"

    "HCA","BX","VRTX","COP","MCK","LMT","PH","KKR","CEG","ADP",
    "CMCSA","CVS","CME","SO","SBUX","HOOD","DUK","BMY","GD","NEM",
    "TT","MMM","MMC","ICE","WM","MCO","ORLY","AMT","SHW","DELL",
    "CDNS","DASH","NOC","REGN","HWM","MAR","TDG","ECL","APO","CTAS",
    "AON","CI","USB","BK","EQIX","MDLZ","PNC","WMB","SNPS","EMR"

    "RCL","ITW","ELV","COR","MNST","JCI","ABNB","GLW","RSG","CL",
    "CMI","AZO","COIN","TRV","AJG","AEP","TEL","NSC","PWR","CSX",
    "HLT","FDX","ADSK","MSI","SRE","WDAY","KMI","SPG","FTNT","TFC",
    "AFL","EOG","IDXX","WBD","MPC","APD","FCX","VST","ROST","ALL",
    "DDOG","BDX","PCAR","SLB","DLR","PSX","ZTS","VLO","D","O"

    "D","O","F","LHX","NDAQ","URI","EA","MET","CAH","EW",
    "BKR","NXPI","XEL","PSA","CBRE","ROP","EXC","DHI","FAST","CARR",
    "LVS","OKE","CMG","CTVA","AME","TTWO","GWW","KR","MPWR","ROK",
    "FANG","ETR","A","FICO","AXON","YUM","MSCI","AMP","DAL","PEG",
    "OXY","AIG","TGT","XYZ","PAYX","CCI","VMC","IQV","HIG","HSY"

    "EQT","KDP","PRU","VTR","CPRT","TRGP","MLM","GRMN","CTSH","EBAY",
    "RMD","NUE","SYY","WEC","GEHC","ED","KMB","WAB","PCG","OTIS",
    "XYL","FIS","ACGL","KEYS","EL","CCL","STT","KVUE","FISV","UAL",
    "NRG","LEN","IR","VRSK","EXPE","VICI","RJF","WTW","LYV","KHC",
    "MTD","ROL","HUM","WRB","FSLR","MTB","ADM","CSGP","K","EXE"

    "IBKR","MCHP","TSCO","HPE","AEE","FITB","ATO","TER","DTE","ODFL",
    "EXR","SYF","FE","EME","BRO","PPL","CBOE","BIIB","BR","CINF",
    "STE","CNP","EFX","CHTR","AVB","IRM","HBAN","DOV","AWK","GIS",
    "ES","VLTO","NTRS","PHM","DXCM","STLD","LDOS","ULTA","DG","WAT",
    "STZ","EQR","TDY","VRSN","DVN","CFG","PODD","CMS","HUBB","HPQ"
]

SYMBOLS = ["QQQ","SPY"]

START_DATE = "2018-01-01"   # backtest start (can adjust)
END_DATE   = "2025-11-16"   # or today's date

OUT_DIR = pathlib.Path("alpha_corp_actions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_SLEEP_SECONDS = 12  # AV free tier: ~5 calls/min; be conservative


# ================== HELPERS ==================
def fetch_daily_adjusted(symbol: str) -> pd.DataFrame:
    """
    Fetch daily adjusted data from Alpha Vantage for a symbol.
    Returns a DataFrame with columns:
      date, open, high, low, close, adj_close, volume,
      dividend, split_coeff
    """
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "datatype": "json",
        "apikey": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    # Basic error / rate-limit handling
    if "Error Message" in data:
        print(f"[ERROR] {symbol}: {data['Error Message']}")
        return pd.DataFrame()

    if "Note" in data:
        print(f"[WARN] {symbol}: Rate limit / note from API: {data['Note']}")
        return pd.DataFrame()

    key = "Time Series (Daily)"
    if key not in data:
        print(f"[ERROR] {symbol}: 'Time Series (Daily)' not in response.")
        return pd.DataFrame()

    ts = data[key]
    rows = []
    for date_str, fields in ts.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "open": float(fields["1. open"]),
            "high": float(fields["2. high"]),
            "low": float(fields["3. low"]),
            "close": float(fields["4. close"]),
            "adj_close": float(fields["5. adjusted close"]),
            "volume": float(fields["6. volume"]),
            "dividend": float(fields["7. dividend amount"]),
            "split_coeff": float(fields["8. split coefficient"]),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    # Restrict to backtest window
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()

    return df


def build_corporate_actions(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Given daily-adjusted df for a symbol, build a corporate actions table:
      - splits (split_coeff != 1.0)
      - potential special dividends (large dividend / price)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol", "date", "action_type", "split_ratio",
            "dividend_amount", "dividend_yield"
        ])

    df = df.copy()
    df["symbol"] = symbol

    # Splits: Alpha's split_coeff is the factor applied to adjust,
    # often 2.0 for 2-for-1; we store it as split_ratio.
    splits = df[df["split_coeff"] != 1.0].copy()
    splits["action_type"] = "split"
    splits["split_ratio"] = splits["split_coeff"]   # you can invert if needed

    # Special dividends: heuristic threshold, e.g. > 2% of price on that day
    df["dividend_yield"] = df["dividend"] / df["adj_close"].replace(0.0, pd.NA)
    specials = df[df["dividend_yield"].fillna(0.0) > 0.02].copy()  # 2% threshold
    specials["action_type"] = "special_dividend"
    specials["split_ratio"] = 1.0

    # Combine
    actions = pd.concat([splits, specials], ignore_index=True, sort=False)
    if actions.empty:
        return pd.DataFrame(columns=[
            "symbol", "date", "action_type", "split_ratio",
            "dividend_amount", "dividend_yield"
        ])

    actions = actions[[
        "symbol", "date", "action_type",
        "split_ratio", "dividend", "dividend_yield"
    ]].rename(columns={"dividend": "dividend_amount"})

    actions = actions.sort_values("date").reset_index(drop=True)
    return actions


# ================== MAIN: RUN FOR ALL SYMBOLS ==================
if __name__ == "__main__":
    all_actions = []

    for sym in SYMBOLS:
        print(f"\n=== {sym}: fetching daily adjusted data ===")
        df = fetch_daily_adjusted(sym)
        if df.empty:
            print(f"[WARN] {sym}: no daily-adjusted data returned.")
        else:
            # Save the raw daily-adjusted series (optional but useful)
            daily_path = OUT_DIR / f"{sym}_daily_adjusted.parquet"
            df.to_parquet(daily_path, index=False)
            print(f"[INFO] {sym}: saved daily-adjusted to {daily_path}")

            # Build corporate actions table
            actions = build_corporate_actions(df, sym)
            if not actions.empty:
                ca_path = OUT_DIR / f"{sym}_corp_actions.parquet"
                actions.to_parquet(ca_path, index=False)
                print(f"[INFO] {sym}: saved {len(actions)} corp actions to {ca_path}")
                all_actions.append(actions)
            else:
                print(f"[INFO] {sym}: no splits/special dividends detected in window.")

        # throttle for Alpha Vantage rate limit
        time.sleep(REQUEST_SLEEP_SECONDS)

    # Global aggregated corporate actions file
    if all_actions:
        all_actions_df = pd.concat(all_actions, ignore_index=True)
        all_actions_path = OUT_DIR / "all_corp_actions.parquet"
        all_actions_df.to_parquet(all_actions_path, index=False)
        print(f"\n[SUMMARY] Saved all corporate actions to {all_actions_path}")
    else:
        print("\n[SUMMARY] No corporate actions detected for any symbol.")