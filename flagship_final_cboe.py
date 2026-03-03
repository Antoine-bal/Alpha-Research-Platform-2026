"""
Flagship Strategy Suite -- CBOE/OptionMetrics Runner.

Runs the 8-strategy suite (5 carry + 3 hedge) on NDX and SPX using
OptionMetrics (WRDS/CBOE) institutional-grade data in om_data/.

Strategy classes imported from flagship_final.py unchanged.

=============================================================================
  Carry: VSS3, VXS3, VCBA, OMDH, SDPS
  Hedge: TH2L, DVMX, XHGE
=============================================================================
Usage:
    python flagship_final_cboe.py --symbol NDX
    python flagship_final_cboe.py --symbol SPX
    python flagship_final_cboe.py --symbol SPX --strat DVMX
=============================================================================

Output: outputs/flagship_suite_cboe/FS_{CODE}_{SYM}.xlsx
"""

import argparse
import dataclasses
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine
from backtester.data_store import DataStore

# Import data store and strategy classes from flagship_final
from flagship_final import (
    _UnifiedDataStore,
    _CARRY_CLASSES, _HEDGE_CLASSES,
    VSS3Strategy, VXS3Strategy, VXCSStrategy,
    OTMCStrategy, SDPSStrategy,
    TH2LStrategy, DVMXStrategy, XHGEStrategy,
)


# ================================================================
# OM DATA STORE -- OptionMetrics column adapter
# ================================================================

class _OMDataStore(_UnifiedDataStore):
    """DataStore for OptionMetrics (WRDS/CBOE) data.

    Handles column mapping from OM conventions to the backtester schema:
      exdate -> expiration, cp_flag -> type, strike_price/1000 -> strike,
      best_bid -> bid, best_offer -> ask, impl_volatility -> implied_volatility,
      optionid -> contractID, (bid+ask)/2 -> mark/last.

    Spot data reads from om_data/spot_data/{SYM}.parquet (close column,
    no split_coeff since NDX/SPX are indices).
    """
    _CACHE_VERSION = "v5_full_om"

    # OM column rename map
    _OM_RENAME = {
        "exdate": "expiration",
        "cp_flag": "type",
        "best_bid": "bid",
        "best_offer": "ask",
        "impl_volatility": "implied_volatility",
        "optionid": "contractID",
    }

    # Columns to read from OM raw parquet
    _OM_READ_COLS = [
        "date", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "volume", "open_interest",
        "impl_volatility", "delta", "gamma", "vega", "theta",
        "optionid",
    ]

    def _load_spot(self):
        """Load spot from om_data/spot_data/{SYM}.parquet.

        OM spot files have: date, close, open, high, low, volume, etc.
        NDX/SPX are indices -- no splits, so split_factor is always 1.0.
        """
        for sym in self.symbols:
            path = self.corp_dir / f"{sym}.parquet"
            if not path.exists():
                print(f"[WARN] OM spot file missing for {sym}: {path}")
                continue

            df = pd.read_parquet(path)
            if "date" not in df.columns:
                print(f"[WARN] No 'date' column in {path}, skipping {sym}")
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
            if df.empty:
                print(f"[WARN] No spot rows for {sym} in backtest window.")
                continue

            df = df.sort_values("date").set_index("date")

            if "close" not in df.columns:
                raise ValueError(f"{sym}: 'close' column missing in {path}")

            close = df["close"].astype(float)
            close = close[close > 0]
            if close.empty:
                print(f"[WARN] No valid close prices for {sym}")
                continue

            out = pd.DataFrame({
                "spot": close,
                "split_factor": 1.0,
            })
            self.spot[sym] = out
            print(f"[INFO] OM spot loaded for {sym}: {out.shape[0]} days, "
                  f"{close.iloc[0]:.0f} -> {close.iloc[-1]:.0f}")

    def _build_optimized_cache(self, sym, raw_path, cache_path,
                               max_dte_cache=9999, min_mny=0.0, max_mny=99.0):
        """Build cache from OM data, mapping columns to backtester schema.

        Streams through OM parquet in chunks, applies column mapping
        (strike_price/1000, rename, compute mid), filters out rows with
        invalid quotes (mark <= 0) and applies moneyness filtering to
        keep cache size manageable for 28M+ row datasets.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        print(f"[CACHE] Building OM cache for {sym} (all DTE, mark>0, mny 0.5-1.5, "
              f"vega*0.01, theta/365)...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Build spot lookup for moneyness filtering
        spot_df = self.spot.get(sym)
        spot_lookup = {}
        if spot_df is not None and not spot_df.empty:
            spot_lookup = spot_df["spot"].to_dict()
            print(f"[CACHE] Spot lookup: {len(spot_lookup)} dates for moneyness filter")

        pf = pq.ParquetFile(raw_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in self._OM_READ_COLS if c in available]
        raw_rows = pf.metadata.num_rows
        print(f"[CACHE] {sym}: {raw_rows:,} OM rows, streaming with column mapping...")

        writer = None
        total_rows = 0
        dropped_mark = 0
        dropped_mny = 0

        for batch in pf.iter_batches(batch_size=500_000, columns=read_cols):
            chunk = pa.Table.from_batches([batch]).to_pandas()
            n_before = len(chunk)

            # strike_price / 1000 (OptionMetrics convention)
            if "strike_price" in chunk.columns:
                chunk["strike"] = chunk["strike_price"].astype(float) / 1000.0
                chunk.drop(columns=["strike_price"], inplace=True)

            # Rename OM -> backtester columns
            chunk.rename(columns=self._OM_RENAME, inplace=True)

            # Parse dates
            chunk["date"] = pd.to_datetime(chunk["date"]).dt.normalize()
            chunk["expiration"] = pd.to_datetime(chunk["expiration"]).dt.normalize()

            # Compute DTE
            chunk["dte"] = (chunk["expiration"] - chunk["date"]).dt.days

            # Normalize type
            chunk["type"] = chunk["type"].astype(str).str.upper().str[0]

            # Compute mid price -> mark and last
            bid = chunk["bid"].astype(float)
            ask = chunk["ask"].astype(float)
            mid = np.where(
                np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0),
                0.5 * (bid + ask),
                np.nan,
            )
            chunk["mark"] = mid
            chunk["last"] = mid

            # OM Greeks convention -> backtester convention:
            #   vega: OM = per 1.0 sigma unit (BS convention) -> * 0.01 = per 1% IV
            #   theta: OM = annualized (dC/dt per year) -> / 365 = daily (calendar-day)
            if "vega" in chunk.columns:
                chunk["vega"] = chunk["vega"].astype(float) * 0.01
            if "theta" in chunk.columns:
                chunk["theta"] = chunk["theta"].astype(float) / 365.0

            # Cast contractID to string
            chunk["contractID"] = chunk["contractID"].astype(str)

            # Filter: DTE >= 0 and mark > 0
            chunk = chunk[(chunk["dte"] >= 0) & (chunk["mark"] > 0)]
            dropped_mark += n_before - len(chunk)
            if chunk.empty:
                del chunk
                continue

            # Moneyness filter using spot data
            if spot_lookup:
                spot_vals = chunk["date"].map(spot_lookup)
                valid_spot = spot_vals.notna() & (spot_vals > 0)
                mny = chunk["strike"] / spot_vals
                keep = ~valid_spot | ((mny >= 0.50) & (mny <= 1.50))
                n_pre_mny = len(chunk)
                chunk = chunk[keep]
                dropped_mny += n_pre_mny - len(chunk)
                if chunk.empty:
                    del chunk
                    continue

            # Keep only columns the backtester expects
            keep_cols = [c for c in DataStore._OPT_COLUMNS if c in chunk.columns]
            chunk = chunk[keep_cols]

            t = pa.Table.from_pandas(chunk, preserve_index=False)
            total_rows += t.num_rows
            if writer is None:
                writer = pq.ParquetWriter(cache_path, t.schema,
                                          use_dictionary=True,
                                          write_statistics=True)
            writer.write_table(t)
            del chunk, t

        if writer is not None:
            writer.close()
        del writer
        gc.collect()

        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        raw_mb = raw_path.stat().st_size / (1024 * 1024)
        print(f"[CACHE] {sym}: {raw_mb:.0f} MB -> {cache_mb:.0f} MB, "
              f"{total_rows:,} rows (dropped {dropped_mark:,} no-quote, "
              f"{dropped_mny:,} out-of-moneyness)")

    def load_symbol_options(self, sym):
        """Load options in yearly chunks to avoid OOM on 20M+ row datasets.

        NDX/SPX have 10x more option rows than QQQ. The parent's
        load_symbol_options tries to process all short-DTE rows (~18M)
        at once, causing OOM in _ingest_options_df. This override reads
        year-by-year with predicate pushdown, processes each chunk, and
        concatenates at the end.
        """
        if sym in self.options:
            return

        import pathlib
        import pyarrow.parquet as pq

        cache_dir = pathlib.Path(self.config.options_cache_dir)
        cache_path = cache_dir / f"{sym}_{self._CACHE_VERSION}.parquet"
        raw_path = self.options_dir / f"{sym}.parquet"

        if not raw_path.exists():
            print(f"[WARN] Options file missing for {sym}: {raw_path}")
            return

        # Build cache if needed
        if not cache_path.exists():
            self._build_optimized_cache(sym, raw_path, cache_path)

        pf = pq.ParquetFile(cache_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in self._OPT_COLUMNS if c in available]

        max_dte_cutoff = max(
            self.config.max_dte_for_entry + 30,
            self.config.rolling_max_dte + 30,
        )

        # Determine year range
        start_year = self.start_date.year
        end_year = self.end_date.year

        all_chunks = []

        # --- Short-DTE: read year by year ---
        SHORT_DTE_MAX = 75
        print(f"  [LOAD] Short-DTE (0-{SHORT_DTE_MAX}), reading by year...")
        for year in range(start_year, end_year + 1):
            yr_start = pd.Timestamp(f"{year}-01-01")
            yr_end = pd.Timestamp(f"{year}-12-31")
            filters = [
                ("date", ">=", yr_start),
                ("date", "<=", yr_end),
                ("dte", ">=", 0),
                ("dte", "<=", SHORT_DTE_MAX),
            ]
            df_chunk = pd.read_parquet(cache_path, columns=read_cols,
                                       filters=filters)
            if df_chunk.empty:
                continue
            self._ingest_options_df(df_chunk, sym)
            del df_chunk
            gc.collect()
            chunk = self.options.pop(sym, pd.DataFrame())
            if not chunk.empty:
                all_chunks.append(chunk)
                print(f"    {year}: {len(chunk):,} rows")
            del chunk
            gc.collect()

        # --- Long-DTE: puts only, year by year ---
        if max_dte_cutoff > SHORT_DTE_MAX:
            LONG_DTE_MIN = SHORT_DTE_MAX + 1
            print(f"  [LOAD] Long-DTE ({LONG_DTE_MIN}-{max_dte_cutoff}, puts only)...")
            for year in range(start_year, end_year + 1):
                yr_start = pd.Timestamp(f"{year}-01-01")
                yr_end = pd.Timestamp(f"{year}-12-31")
                filters = [
                    ("date", ">=", yr_start),
                    ("date", "<=", yr_end),
                    ("dte", ">=", LONG_DTE_MIN),
                    ("dte", "<=", max_dte_cutoff),
                    ("type", "==", "P"),
                ]
                df_chunk = pd.read_parquet(cache_path, columns=read_cols,
                                           filters=filters)
                if df_chunk.empty:
                    continue
                self._ingest_options_df(df_chunk, sym)
                del df_chunk
                gc.collect()
                chunk = self.options.pop(sym, pd.DataFrame())
                if not chunk.empty:
                    all_chunks.append(chunk)
                    print(f"    {year}: {len(chunk):,} rows (long-DTE puts)")
                del chunk
                gc.collect()

        if all_chunks:
            combined = pd.concat(all_chunks, ignore_index=True)
            del all_chunks
            gc.collect()
            combined.sort_values("date", inplace=True, ignore_index=True)
            combined.set_index("date", drop=False, inplace=True)
            self.options[sym] = combined
            print(f"  [LOAD] Total: {len(combined):,} rows")
        else:
            self.options[sym] = pd.DataFrame()
            print(f"  [LOAD] No data loaded for {sym}")


# ================================================================
# STRATEGY SUITE DEFINITIONS (8 configs)
# ================================================================

OUTPUT_DIR = Path("outputs/flagship_suite_cboe")


def _make_config(code, cls, delta_hedge=False, symbol="NDX", max_dte=45, reinvest=True,
                  base_vega_target=0.5, execution_mode="mid_spread"):
    mode = f"fs_{cls._CODE.lower()}"
    cfg = BacktestConfig(
        symbols=[symbol],
        start_date=pd.Timestamp("2016-01-04"),
        end_date=pd.Timestamp("2025-08-29"),
        strategy_mode=mode,
        initial_perf_per_ticker=100.0,
        reinvest=reinvest,
        base_vega_target=base_vega_target,
        delta_hedge=delta_hedge,
        execution_mode=execution_mode,
        cost_model={"option_spread_bps": 50, "stock_spread_bps": 1,
                    "commission_per_contract": 0.0},
        min_moneyness=0.5,
        max_moneyness=1.5,
        min_dte_for_entry=1,
        max_dte_for_entry=max_dte,
        rolling_max_dte=max_dte,
        optimized_options_loading=True,
        options_dir="om_data/opt_raw",
        corp_dir="om_data/spot_data",
        options_cache_dir="cache_options",
        output_path=str(OUTPUT_DIR / f"FS_{code}_{symbol}.xlsx"),
        pnl_explain=True,
        exit_fallback_mode="intrinsic",
        trade_log_mode="light",
    )
    cat = "carry" if cls in _CARRY_CLASSES else "hedge"
    return {"name": f"FS_{code}", "config": cfg, "code": code, "category": cat}


def _build_strategies(symbol="NDX"):
    return [
        # Carry (5 strategies)
        _make_config("VSS3", VSS3Strategy, delta_hedge=True, symbol=symbol, max_dte=14, base_vega_target=1.5),
        _make_config("VXS3", VXS3Strategy, delta_hedge=True, symbol=symbol, base_vega_target=0.25),
        _make_config("VCBA", VXCSStrategy, delta_hedge=True, symbol=symbol, base_vega_target=0.25,
                     execution_mode="bid_ask"),
        _make_config("OMDH", OTMCStrategy, delta_hedge=True, symbol=symbol, base_vega_target=1.0),
        _make_config("SDPS", SDPSStrategy, delta_hedge=False, symbol=symbol, base_vega_target=1.5),
        # Hedge (3 strategies, reinvest=False -- constant sizing)
        _make_config("TH2L", TH2LStrategy, delta_hedge=True, symbol=symbol, max_dte=500, reinvest=False, base_vega_target=1.0),
        _make_config("DVMX", DVMXStrategy, delta_hedge=True, symbol=symbol, max_dte=500, reinvest=False, base_vega_target=1.0),
        _make_config("XHGE", XHGEStrategy, delta_hedge=True, symbol=symbol, reinvest=False, base_vega_target=1.0),
    ]


STRATEGIES = _build_strategies("NDX")


# ================================================================
# RUNNER
# ================================================================

def run_cboe_suite(strat_filter=None, symbol="NDX"):
    """Run the flagship final suite on OM/CBOE data."""
    global STRATEGIES
    STRATEGIES = _build_strategies(symbol)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if strat_filter:
        to_run = [s for s in STRATEGIES if s["code"] in strat_filter]
    else:
        to_run = STRATEGIES

    if not to_run:
        print("No strategies to run!")
        return

    print(f"\n{'=' * 60}")
    print(f"FLAGSHIP FINAL SUITE (CBOE/OM) -- {len(to_run)} configs on {symbol}")
    print(f"{'=' * 60}")

    # Compute shared config for data loading
    max_dte_needed = max(s["config"].max_dte_for_entry for s in to_run)
    max_rolling_dte = max(s["config"].rolling_max_dte for s in to_run)
    load_cfg = dataclasses.replace(
        to_run[0]["config"],
        max_dte_for_entry=max_dte_needed,
        rolling_max_dte=max_rolling_dte,
        optimized_options_loading=False,
    )

    print(f"\nLoading {symbol} OM data (max_dte={max_dte_needed})...")
    shared_store = _OMDataStore(load_cfg)
    shared_store.load_symbol_options(symbol)
    print(f"Options loaded: {shared_store.options.get(symbol, pd.DataFrame()).shape[0]:,} rows")

    results = []
    for i, strat in enumerate(to_run):
        code = strat["code"]
        cat = strat["category"]
        dh = "DH" if strat["config"].delta_hedge else "noDH"
        print(f"\n[{i + 1}/{len(to_run)}] Running {code} ({cat}, {dh})...")
        t0 = time.time()

        try:
            engine = BacktestEngine(strat["config"], market=shared_store)
            metrics = engine.run()
            elapsed = time.time() - t0

            if metrics:
                m = metrics
                sharpe = getattr(m, "sharpe_ratio", np.nan)
                ann_ret = getattr(m, "annualized_return", np.nan)
                ann_vol = getattr(m, "annualized_vol", np.nan)
                max_dd = getattr(m, "max_drawdown", np.nan)
                total_ret = getattr(m, "total_return", np.nan)
                results.append({
                    "code": code, "category": cat,
                    "dh": strat["config"].delta_hedge,
                    "total_return": total_ret, "ann_return": ann_ret,
                    "ann_vol": ann_vol, "sharpe": sharpe,
                    "max_dd": max_dd, "time_s": round(elapsed, 1),
                })
                sr_str = f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A"
                ret_str = f"{ann_ret:.1%}" if np.isfinite(ann_ret) else "N/A"
                vol_str = f"{ann_vol:.1%}" if np.isfinite(ann_vol) else "N/A"
                dd_str = f"{max_dd:.1%}" if np.isfinite(max_dd) else "N/A"
                print(f"  Sharpe={sr_str}  Return={ret_str}  Vol={vol_str}  MaxDD={dd_str}  ({elapsed:.1f}s)")
            else:
                print(f"  No metrics returned ({elapsed:.1f}s)")
                results.append({"code": code, "category": cat, "time_s": round(elapsed, 1)})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e} ({elapsed:.1f}s)")
            import traceback
            traceback.print_exc()
            results.append({"code": code, "category": cat, "error": str(e)})

    # Summary table
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Code':8s} {'Category':10s} {'Return':>8s} {'Vol':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'DH':>4s}")
    print("-" * 75)
    dh_map = {s["code"]: s["config"].delta_hedge for s in STRATEGIES}
    for r in results:
        code = r.get("code", "?")
        cat = r.get("category", "?")
        ann_r = r.get("ann_return", np.nan)
        ann_v = r.get("ann_vol", np.nan)
        sr = r.get("sharpe", np.nan)
        dd = r.get("max_dd", np.nan)
        dh = "Y" if dh_map.get(code, False) else "N"
        if all(np.isfinite([ann_r, ann_v, sr, dd])):
            print(f"{code:8s} {cat:10s} {ann_r:>+7.1%} {ann_v:>7.1%} {sr:>7.2f} {dd:>+7.1%}  {dh:>2s}")
        else:
            err = r.get("error", "N/A")
            print(f"{code:8s} {cat:10s} {'ERR':>8s} {'':>8s} {'':>8s} {'':>8s}  {dh:>2s}  {err[:40]}")

    summary_df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / f"final_summary_{symbol}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flagship Final Suite (CBOE/OM)")
    parser.add_argument("--carry-only", action="store_true")
    parser.add_argument("--hedge-only", action="store_true")
    parser.add_argument("--strat", type=str, help="Run single strategy by code (or comma-separated)")
    parser.add_argument("--symbol", type=str, default="NDX",
                        help="Underlying symbol: NDX or SPX (default: NDX)")
    args = parser.parse_args()

    sym = args.symbol.upper()
    STRATEGIES = _build_strategies(sym)

    if args.strat:
        filt = [s.strip().upper() for s in args.strat.split(",")]
    elif args.carry_only:
        filt = [s["code"] for s in STRATEGIES if s["category"] == "carry"]
    elif args.hedge_only:
        filt = [s["code"] for s in STRATEGIES if s["category"] == "hedge"]
    else:
        filt = None

    run_cboe_suite(filt, symbol=sym)
