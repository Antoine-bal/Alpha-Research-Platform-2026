"""
Flagship Strategy Suite -- 8 Strategies (5 Carry + 3 Hedge).

=============================================================================
CARRY (5) -- short vol, reinvested
=============================================================================
VSS3   Short varswap strip (1-20 delta), MWF ladder, DH
VXS3   VIX-implied 3.0-sigma OTM put selling, daily, DH
VCBA   VIX-implied 1.5-sigma OTM call selling, daily, DH, bid/ask execution
OMDH   OTM call selling (all calls <=10 delta, <=7D), daily, DH
SDPS   Bull put spread -20D/-5D, 7D, MWF, signal-gated, no DH

=============================================================================
HEDGE (3) -- long vol/convexity, constant sizing
=============================================================================
TH2L   Theta-neutral puts (long 5D / short 15D), 2x long leg, 3x vega, 55D hold, DH
DVMX   Down-variance swap (60-85% puts, 2.5% grid), max(skew, 20% overhedge) DH
XHGE   Macro-triggered long ATM straddle (>=2/4 crisis triggers), DH

=============================================================================
Usage:
    python flagship_final.py                         # all 8, QQQ
    python flagship_final.py --symbol SPY             # all 8, SPY
    python flagship_final.py --strat DVMX --symbol SPY
    python flagship_final.py --carry-only
    python flagship_final.py --hedge-only
=============================================================================

Output: outputs/flagship_final/FS_{CODE}_{SYMBOL}.xlsx
"""

import argparse
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from backtester.config import BacktestConfig
from backtester.engine import BacktestEngine, STRATEGY_REGISTRY
from backtester.data_store import DataStore
from backtester.strategies.base import Strategy


class _UnifiedDataStore(DataStore):
    """DataStore with a single full cache containing ALL raw options data.

    Builds one cache file (QQQ_v4_full.parquet) with ALL DTE/types.
    Loads it in two memory-safe predicate-pushdown reads:
      1. Short-DTE (0-75): all types -- for most strategies
      2. Long-DTE (60-430): puts only -- for DVAR/DVAS
    Merges both into a single options DataFrame.
    """
    _CACHE_VERSION = "v4_full"

    def _build_optimized_cache(self, sym, raw_path, cache_path,
                               max_dte_cache=9999, min_mny=0.0, max_mny=99.0):
        """Build cache with ALL data, chunked write without in-memory sort."""
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq

        print(f"[CACHE] Building full cache for {sym} (all DTE, all types)...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        pf = pq.ParquetFile(raw_path)
        available = set(pf.schema_arrow.names)
        base_cols = [c for c in DataStore._OPT_COLUMNS if c in available and c != "dte"]
        raw_rows = pf.metadata.num_rows
        print(f"[CACHE] {sym}: {raw_rows:,} raw rows, streaming to cache...")

        writer = None
        total_rows = 0

        for batch in pf.iter_batches(batch_size=500_000, columns=base_cols):
            chunk = pa.Table.from_batches([batch]).to_pandas()

            # Compute DTE
            chunk["date"] = pd.to_datetime(chunk["date"]).dt.normalize()
            chunk["expiration"] = pd.to_datetime(chunk["expiration"]).dt.normalize()
            chunk["dte"] = (chunk["expiration"] - chunk["date"]).dt.days
            chunk["strike"] = chunk["strike"].astype(float)
            chunk["type"] = chunk["type"].astype(str).str.upper().str[0]

            # Minimal filter: DTE >= 0 only
            chunk = chunk[chunk["dte"] >= 0]
            if chunk.empty:
                del chunk
                continue

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
              f"{total_rows:,} rows (full, unsorted)")

    def load_symbol_options(self, sym):
        """Load options in two memory-safe reads from the single full cache."""
        if sym in self.options:
            return

        import gc
        import pathlib
        import pyarrow.parquet as pq

        cache_dir = pathlib.Path(self.config.options_cache_dir)
        cache_path = cache_dir / f"{sym}_{self._CACHE_VERSION}.parquet"
        raw_path = self.options_dir / f"{sym}.parquet"

        if not raw_path.exists():
            print(f"[WARN] Options file missing for {sym}: {raw_path}")
            return

        # Build full cache if it doesn't exist
        if not cache_path.exists():
            self._build_optimized_cache(sym, raw_path, cache_path)

        pf = pq.ParquetFile(cache_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in self._OPT_COLUMNS if c in available]

        date_filters = [
            ("date", ">=", self.start_date),
            ("date", "<=", self.end_date),
        ]

        # --- Read 1: short-DTE (0-75), all types ---
        SHORT_DTE_MAX = 75
        filters_short = date_filters + [("dte", ">=", 0), ("dte", "<=", SHORT_DTE_MAX)]
        print(f"  [LOAD] Read 1: short-DTE (0-{SHORT_DTE_MAX}, all types)...")
        df_short = pd.read_parquet(cache_path, columns=read_cols, filters=filters_short)
        print(f"  [LOAD]   -> {len(df_short):,} rows")

        # Process short-DTE data
        self._ingest_options_df(df_short, sym)
        del df_short
        gc.collect()

        short_opts = self.options.pop(sym, pd.DataFrame())
        print(f"  [LOAD]   -> {len(short_opts):,} rows after processing")

        # --- Read 2: long-DTE (60-430), puts only ---
        LONG_DTE_MIN = SHORT_DTE_MAX + 1  # avoid overlap
        max_dte_cutoff = max(
            self.config.max_dte_for_entry + 30,
            self.config.rolling_max_dte + 30,
        )
        if max_dte_cutoff > SHORT_DTE_MAX:
            filters_long = date_filters + [
                ("dte", ">=", LONG_DTE_MIN),
                ("dte", "<=", max_dte_cutoff),
                ("type", "==", "P"),
            ]
            print(f"  [LOAD] Read 2: long-DTE ({LONG_DTE_MIN}-{max_dte_cutoff}, puts only)...")
            df_long = pd.read_parquet(cache_path, columns=read_cols, filters=filters_long)
            print(f"  [LOAD]   -> {len(df_long):,} rows")

            if not df_long.empty:
                self._ingest_options_df(df_long, sym)
                del df_long
                gc.collect()

                long_opts = self.options.pop(sym, pd.DataFrame())
                print(f"  [LOAD]   -> {len(long_opts):,} rows after processing")

                # Merge
                short_opts = pd.concat([short_opts, long_opts], ignore_index=True)
                short_opts.sort_values("date", inplace=True, ignore_index=True)
                del long_opts
                gc.collect()
            else:
                del df_long

        # Restore date index required by get_chain (df.loc[date])
        if "date" in short_opts.columns:
            short_opts = short_opts.set_index("date", drop=False)
        self.options[sym] = short_opts
        print(f"  [LOAD] Total: {len(self.options[sym]):,} rows")

from flagship_signals import (
    FlagshipSignalMixin, _load_vol_surface,
    compute_gex, compute_bf25, compute_chain_skew, compute_atm_iv,
)
from har_rv import HARRVModel


# ================================================================
# OPTION SELECTION HELPERS
# ================================================================

def _ensure_chain(chain, date):
    if chain is None or chain.empty:
        return None
    if "expiration" not in chain.columns:
        if "expiry" in chain.columns:
            chain = chain.rename(columns={"expiry": "expiration"})
        else:
            return None
    if "dte" not in chain.columns:
        chain = chain.copy()
        chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
        chain["dte"] = (chain["expiration"] - date.normalize()).dt.days
    return chain


def _get_expiry(chain, target_dte, min_dte=1, max_dte=60):
    eligible = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]
    if eligible.empty:
        return None
    exp_dte = eligible.groupby("expiration")["dte"].min()
    cands = exp_dte[exp_dte >= target_dte]
    if cands.empty:
        cands = exp_dte[exp_dte >= min_dte]
    if cands.empty:
        return None
    return pd.to_datetime(cands.idxmin()).normalize()


def _get_expiry_nearest_date(chain, target_date):
    """Find the chain expiry closest to a specific target date."""
    expirations = pd.to_datetime(chain["expiration"].unique())
    if len(expirations) == 0:
        return None
    diffs = np.abs((expirations - target_date).total_seconds())
    return expirations[np.argmin(diffs)].normalize()


def _select_delta(sub, opt_type, target_delta):
    opts = sub[sub["type"] == opt_type]
    if opts.empty or "delta" not in opts.columns:
        return None
    delta = opts["delta"].astype(float)
    valid = opts[np.isfinite(delta)]
    if valid.empty:
        return None
    pos = (valid["delta"].astype(float) - target_delta).abs().values.argmin()
    row = valid.iloc[pos]
    mid = float(row.get("mid", row.get("mark", 0)))
    if mid <= 0:
        bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
        mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0
    if mid <= 0:
        return None
    vega = float(row.get("vega", 0))
    if not np.isfinite(vega) or abs(vega) < 1e-12:
        return None
    return row


def _select_atm(sub, opt_type, spot):
    opts = sub[sub["type"] == opt_type]
    if opts.empty:
        return None
    mny = (opts["strike"].astype(float) / spot - 1.0).abs()
    valid = opts[mny < 0.05]
    if valid.empty:
        valid = opts
    pos = (valid["strike"].astype(float) / spot - 1.0).abs().values.argmin()
    row = valid.iloc[pos]
    mid = float(row.get("mid", row.get("mark", 0)))
    if mid <= 0:
        bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
        mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0
    if mid <= 0:
        return None
    return row


def _select_strike(sub, opt_type, target_strike):
    """Select nearest option to a specific strike price."""
    opts = sub[sub["type"] == opt_type]
    if opts.empty:
        return None
    strike_diff = (opts["strike"].astype(float) - target_strike).abs()
    at_or_below = opts[opts["strike"].astype(float) <= target_strike]
    if not at_or_below.empty:
        pos = (at_or_below["strike"].astype(float) - target_strike).abs().values.argmin()
        row = at_or_below.iloc[pos]
    else:
        pos = strike_diff.values.argmin()
        row = opts.iloc[pos]
    mid = float(row.get("mid", row.get("mark", 0)))
    if mid <= 0:
        bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
        mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0
    if mid <= 0:
        return None
    return row


def _leg(row, expiry, qty):
    return {
        "contract_id": str(row.get("contractID", "")),
        "expiry": expiry,
        "strike": float(row["strike"]),
        "type": str(row["type"]).upper()[0],
        "qty": float(qty),
    }


def _is_entry_day(date, frequency="weekly", weekday=4):
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return date.weekday() == weekday
    if frequency == "mwf":
        return date.weekday() in (0, 2, 4)
    if frequency == "biweekly":
        return date.weekday() == weekday and (date.isocalendar()[1] % 2 == 0)
    if frequency == "monthly":
        if date.weekday() != 4:
            return False
        return 15 <= date.day <= 21
    return False


def _row_vega(row):
    v = float(row.get("vega", 0))
    return abs(v) if np.isfinite(v) else 0.0


def _row_mid(row):
    mid = float(row.get("mid", row.get("mark", 0)))
    if mid <= 0:
        bid, ask = float(row.get("bid", 0)), float(row.get("ask", 0))
        mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0
    return mid


# ================================================================
# STRUCTURE BUILDERS
# ================================================================

def _build_straddle(chain, date, spot, target_dte, min_dte=21, max_dte=45):
    expiry = _get_expiry(chain, target_dte, min_dte, max_dte)
    if expiry is None:
        return None
    sub = chain[chain["expiration"] == expiry]
    call = _select_atm(sub, "C", spot)
    put = _select_atm(sub, "P", spot)
    if call is None or put is None:
        return None
    if _row_vega(call) < 1e-12 or _row_vega(put) < 1e-12:
        return None
    return (expiry, call, put)


def _build_put_spread(chain, date, spot, short_delta, long_delta, target_dte,
                       min_dte=21, max_dte=45):
    expiry = _get_expiry(chain, target_dte, min_dte, max_dte)
    if expiry is None:
        return None
    sub = chain[chain["expiration"] == expiry]
    short_put = _select_delta(sub, "P", short_delta)
    long_put = _select_delta(sub, "P", long_delta)
    if short_put is None or long_put is None:
        return None
    if float(long_put["strike"]) >= float(short_put["strike"]):
        return None
    return (expiry, short_put, long_put)


def _build_varswap_strip(chain, date, spot, target_dte, delta_cutoff=0.05,
                          delta_max=0.50, direction=-1, min_dte=3, max_dte=14):
    """Build OTM variance swap strip with 2*dK/K^2 weights (Demeterfi et al. 1999).

    Returns (expiry, strip_df, weights, weighted_vega) or None.
    Minimum strip size: 4 options.
    """
    expiry = _get_expiry(chain, target_dte, min_dte, max_dte)
    if expiry is None:
        return None
    sub = chain[chain["expiration"] == expiry].copy()
    if sub.empty or "delta" not in sub.columns:
        return None

    # OTM puts: K < spot
    puts = sub[
        (sub["type"] == "P")
        & (sub["strike"] < spot)
        & (sub["delta"].astype(float).abs() >= delta_cutoff)
        & (sub["delta"].astype(float).abs() <= delta_max)
    ]
    # OTM calls: K >= spot
    calls = sub[
        (sub["type"] == "C")
        & (sub["strike"] >= spot)
        & (sub["delta"].astype(float).abs() >= delta_cutoff)
        & (sub["delta"].astype(float).abs() <= delta_max)
    ]

    strip = pd.concat([puts, calls], ignore_index=True)
    strip = strip.sort_values("strike").reset_index(drop=True)

    # Filter for valid vega and mid
    valid_mask = strip["vega"].astype(float).abs() > 1e-12
    strip = strip[valid_mask].reset_index(drop=True)
    if len(strip) < 4:
        return None

    # Filter for valid mid prices
    mids = strip.apply(_row_mid, axis=1)
    strip = strip[mids > 0].reset_index(drop=True)
    if len(strip) < 4:
        return None

    # Compute dK (strike spacing: central difference interior, one-sided edges)
    strikes = strip["strike"].values.astype(float)
    n = len(strikes)
    dk = np.zeros(n)
    for i in range(n):
        if i == 0:
            dk[i] = strikes[1] - strikes[0]
        elif i == n - 1:
            dk[i] = strikes[-1] - strikes[-2]
        else:
            dk[i] = (strikes[i + 1] - strikes[i - 1]) / 2.0

    # Weights: 2 * dK / K^2
    weights = 2.0 * dk / (strikes ** 2)

    vegas = strip["vega"].values.astype(float)
    weighted_vega = np.sum(weights * np.abs(vegas))
    if weighted_vega < 1e-12:
        return None

    return (expiry, strip, weights, weighted_vega)


def _compute_semi_annual_expiry(date, n=3):
    """Compute nth semi-annual expiry (3rd Friday of Jun/Dec) strictly after date."""
    candidates = []
    for year_offset in range(4):
        year = date.year + year_offset
        for month in [6, 12]:
            d = pd.Timestamp(year, month, 1)
            days_to_fri = (4 - d.weekday()) % 7
            first_friday = d + pd.Timedelta(days=days_to_fri)
            third_friday = first_friday + pd.Timedelta(days=14)
            if third_friday.normalize() > date.normalize():
                candidates.append(third_friday.normalize())
    candidates.sort()
    if len(candidates) >= n:
        return candidates[n - 1]
    return candidates[-1] if candidates else None


def _compute_nth_third_friday(date, n=4):
    """Compute nth monthly 3rd-Friday expiry strictly after date."""
    candidates = []
    for m_offset in range(n + 6):
        year = date.year + (date.month - 1 + m_offset + 1) // 12
        month = (date.month + m_offset) % 12 + 1
        d = pd.Timestamp(year, month, 1)
        days_to_fri = (4 - d.weekday()) % 7
        first_friday = d + pd.Timedelta(days=days_to_fri)
        third_friday = first_friday + pd.Timedelta(days=14)
        if third_friday.normalize() > date.normalize():
            candidates.append(third_friday.normalize())
    candidates.sort()
    if len(candidates) >= n:
        return candidates[n - 1]
    return candidates[-1] if candidates else None


def _build_down_varswap_strip(chain, date, spot, target_dte,
                                moneyness_levels=None, min_dte=90, max_dte=400):
    """Build downside variance swap strip: long puts at specific moneyness levels.

    Direction = LONG variance (buy puts = long downside vol).
    Returns (expiry, rows, weights, weighted_vega) or None.
    """
    if moneyness_levels is None:
        moneyness_levels = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    expiry = _get_expiry(chain, target_dte, min_dte, max_dte)
    if expiry is None:
        return None
    sub = chain[chain["expiration"] == expiry]

    rows = []
    for m in sorted(moneyness_levels):
        target_strike = spot * m
        row = _select_strike(sub, "P", target_strike)
        if row is not None and _row_vega(row) > 1e-12:
            rows.append(row)

    if len(rows) < 4:
        return None

    # Sort by strike ascending
    strikes = np.array([float(r["strike"]) for r in rows])
    sort_idx = np.argsort(strikes)
    rows = [rows[i] for i in sort_idx]
    strikes = strikes[sort_idx]

    n = len(strikes)
    dk = np.zeros(n)
    for i in range(n):
        if i == 0:
            dk[i] = strikes[1] - strikes[0]
        elif i == n - 1:
            dk[i] = strikes[-1] - strikes[-2]
        else:
            dk[i] = (strikes[i + 1] - strikes[i - 1]) / 2.0

    weights = 2.0 * dk / (strikes ** 2)

    vegas = np.array([_row_vega(r) for r in rows])
    weighted_vega = np.sum(weights * vegas)
    if weighted_vega < 1e-12:
        return None

    return (expiry, rows, weights, weighted_vega)


def _build_theta_neutral_puts(chain, date, spot, target_dte=365,
                               buy_delta=-0.02, sell_delta=-0.10,
                               min_dte=180, max_dte=450,
                               max_ratio=3.0, expiry_override=None):
    """Build theta-neutral put structure for tail hedging.

    Buy far OTM (2D) puts, sell closer OTM (10D) puts.
    Sizing ratio: sell_qty/buy_qty = |buy_theta|/|sell_theta| so net theta ~ 0.
    max_ratio caps the sell/buy ratio (None = no cap).
    expiry_override skips DTE-based selection and uses the given expiry directly.
    Returns (expiry, buy_row, sell_row, sell_qty_ratio) or None.
    """
    if expiry_override is not None:
        expiry = expiry_override
    else:
        expiry = _get_expiry(chain, target_dte, min_dte, max_dte)
    if expiry is None:
        return None
    sub = chain[chain["expiration"] == expiry]

    buy_row = _select_delta(sub, "P", buy_delta)    # deep OTM (2D)
    sell_row = _select_delta(sub, "P", sell_delta)   # closer OTM (10D)
    if buy_row is None or sell_row is None:
        return None

    # buy_row (2D) should have LOWER strike (deeper OTM) than sell_row (10D)
    if float(buy_row["strike"]) >= float(sell_row["strike"]):
        return None

    # Theta-neutral ratio
    buy_theta = float(buy_row.get("theta", 0))
    sell_theta = float(sell_row.get("theta", 0))

    if (np.isfinite(buy_theta) and np.isfinite(sell_theta)
            and abs(sell_theta) > 1e-12 and abs(buy_theta) > 1e-12):
        sell_qty_ratio = abs(buy_theta) / abs(sell_theta)
    else:
        # Fallback: use vega ratio as proxy for theta ratio
        buy_v = _row_vega(buy_row)
        sell_v = _row_vega(sell_row)
        if sell_v > 1e-12:
            sell_qty_ratio = buy_v / sell_v
        else:
            return None

    if max_ratio is not None:
        sell_qty_ratio = min(sell_qty_ratio, max_ratio)
    return (expiry, buy_row, sell_row, sell_qty_ratio)


def _compute_backfill_schedule(backtest_start, frequency, hold_months,
                                backfill_start=pd.Timestamp("2019-01-01")):
    """Compute positions that would still be open at backtest_start.

    Returns list of (orig_entry, orig_exit, days_elapsed) for each theoretical
    entry from backfill_start whose exit_date > backtest_start.
    """
    entries = []
    bs_norm = backtest_start.normalize()
    current = backfill_start.normalize()
    while current < bs_norm:
        if _is_entry_day(current, frequency):
            exit_dt = (current + pd.DateOffset(months=hold_months)).normalize()
            if exit_dt > bs_norm:
                days_elapsed = (bs_norm - current).days
                entries.append((current, exit_dt, days_elapsed))
        current += pd.Timedelta(days=1)
    return entries


def _next_mwf_dates(date, n=3):
    """Return the next n MWF calendar dates strictly after `date`."""
    mwf = {0, 2, 4}  # Monday, Wednesday, Friday
    result = []
    d = date + pd.Timedelta(days=1)
    while len(result) < n:
        if d.weekday() in mwf:
            result.append(d)
        d += pd.Timedelta(days=1)
    return result


# ================================================================
# SIGNAL TRANSFORM HELPERS
# ================================================================

def _zscore_pos(x):
    return np.clip(x / 2.0, -1.0, 1.0)

def _zscore_neg(x):
    return np.clip(-x / 2.0, -1.0, 1.0)

def _regime_pos(x):
    return np.clip(x, -1.0, 1.0)

def _rv_ratio_low(x):
    return np.clip((1.0 - x) * 3.0, -1.0, 1.0)


# ================================================================
# BASE STRATEGY CLASS
# ================================================================

class _BaseStrategy(FlagshipSignalMixin, Strategy):
    """Common base for all strategies with profit-taking and stop-loss."""

    _CODE = ""
    _SIGNAL_CONFIG = {}
    _PROFIT_TARGET = 0.50
    _BASE_MULT = 0.10
    _FREQUENCY = "weekly"
    _WEEKDAY = 4

    def initialize(self, market):
        self.signals = self._build_flagship_signals(market)
        self._load_vol_surface()
        self._iv_history = {}

    def _check_early_exits(self, date, symbol, state, market):
        """Check live positions for profit-taking and stop-loss."""
        if self._PROFIT_TARGET is None and not any(
            ptf.meta.get("loss_threshold") is not None
            for ptf in state.rolling_ptfs.values()
        ):
            return

        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return

        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None or chain.empty:
            return

        if "contractID" not in chain.columns:
            return

        # Collect only the contract IDs we actually need
        needed_cids = set()
        for ptf in state.rolling_ptfs.values():
            if date <= ptf.entry_date or date >= ptf.exit_date:
                continue
            for leg_obj in ptf.legs:
                if abs(leg_obj.qty) >= 1e-12:
                    needed_cids.add(leg_obj.contract_id)
        if not needed_cids:
            return

        # Fast vectorized lookup
        cid_col = chain["contractID"].astype(str)
        mask = cid_col.isin(needed_cids)
        subset = chain.loc[mask]
        chain_lookup = {}
        if "mid" in subset.columns:
            chain_lookup = dict(zip(cid_col[mask], subset["mid"]))
        else:
            chain_lookup = dict(zip(
                cid_col[mask],
                (subset["bid"] + subset["ask"]) / 2,
            ))

        for ptf_id, ptf in list(state.rolling_ptfs.items()):
            if date <= ptf.entry_date or date >= ptf.exit_date:
                continue

            entry_credit = ptf.meta.get("entry_credit")
            profit_target = ptf.meta.get("profit_target", self._PROFIT_TARGET)
            loss_threshold = ptf.meta.get("loss_threshold")

            if entry_credit is None or not np.isfinite(entry_credit) or entry_credit <= 0:
                continue

            current_cost = 0.0
            any_priced = False
            for leg_obj in ptf.legs:
                if abs(leg_obj.qty) < 1e-12:
                    continue
                mid = chain_lookup.get(leg_obj.contract_id, 0.0)
                if mid > 0:
                    current_cost += -leg_obj.qty * mid
                    any_priced = True

            if not any_priced:
                continue

            profit = entry_credit - current_cost
            profit_pct = profit / entry_credit if entry_credit > 0 else 0

            if profit_target is not None and profit_pct >= profit_target:
                ptf.exit_date = date
                ptf.meta["early_exit"] = True
                ptf.meta["exit_reason"] = "profit_target"
                ptf.meta["exit_profit_pct"] = round(profit_pct, 3)
                continue

            if loss_threshold is not None and profit_pct < loss_threshold:
                ptf.exit_date = date
                ptf.meta["early_exit"] = True
                ptf.meta["exit_reason"] = "stop_loss"
                ptf.meta["exit_profit_pct"] = round(profit_pct, 3)

    def _compute_entry_credit(self, legs_data):
        """Compute net entry credit from leg list.
        Returns positive for net credit (short premium), negative for net debit."""
        total = 0.0
        for row, qty in legs_data:
            mid = _row_mid(row)
            total += -qty * mid
        return total


# ================================================================
# CARRY STRATEGIES
# ================================================================

class _VarswapBaseStrategy(_BaseStrategy):
    """Base variance swap: full OTM strip with 2*dK/K^2 weights.

    Replicates a short variance swap using the Demeterfi et al. (1999) method.
    Weekly Friday entry, 7D DTE, delta-hedged. No signal scaling.
    """
    _CODE = "VSWA"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.10
    _FREQUENCY = "weekly"
    _WEEKDAY = 4
    _SIGNAL_CONFIG = {}

    # Subclass can override these for filtered variants
    _DELTA_CUTOFF = 0.05
    _DELTA_MAX = 0.50

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY, self._WEEKDAY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        scale = self._signal_scale(symbol, date, self._SIGNAL_CONFIG) if self._SIGNAL_CONFIG else 1.0

        result = _build_varswap_strip(
            chain, date, spot, target_dte=7,
            delta_cutoff=self._DELTA_CUTOFF, delta_max=self._DELTA_MAX,
            direction=-1, min_dte=3, max_dte=14,
        )
        if result is None:
            return
        expiry, strip, weights, weighted_vega = result

        n = len(strip)
        legs = []
        legs_data = []
        for i in range(n):
            row = strip.iloc[i]
            qty = -1.0 * abs(vega_target) * self._BASE_MULT * scale * weights[i] / weighted_vega
            if abs(qty) < 1e-10:
                continue
            legs.append(_leg(row, expiry, qty))
            legs_data.append((row, qty))

        if not legs:
            return

        entry_credit = self._compute_entry_credit(legs_data)
        portfolio.register_new_ptf(symbol, date.normalize(), expiry, legs,
                                    {"mode": self._CODE, "scale": round(scale, 2),
                                     "entry_credit": entry_credit,
                                     "n_legs": len(legs),
                                     "structure": "varswap_strip"})


class _MWFLadderStrategy(_VarswapBaseStrategy):
    """MWF varswap ladder: full OTM strip, 3 buckets targeting next 3 MWF expiries.

    Bucket allocation: 5/9 (front), 3/9 (mid), 1/9 (back).
    No signal scaling. Delta-hedged.
    """
    _CODE = "VSS1"
    _DELTA_CUTOFF = 0.05
    _DELTA_MAX = 0.50
    _SIGNAL_CONFIG = {}
    _BUCKET_WEIGHTS = (5.0 / 9.0, 3.0 / 9.0, 1.0 / 9.0)
    _FREQUENCY = "mwf"
    _BASE_MULT = 0.10 / 3.0    # /3 vs base: MWF = 3x weekly frequency

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        targets = _next_mwf_dates(date, n=3)

        for i, target_date in enumerate(targets):
            dte_target = (target_date - date).days
            min_dte = max(1, dte_target - 1)
            max_dte = dte_target + 1

            result = _build_varswap_strip(
                chain, date, spot, target_dte=dte_target,
                delta_cutoff=self._DELTA_CUTOFF, delta_max=self._DELTA_MAX,
                direction=-1, min_dte=min_dte, max_dte=max_dte,
            )
            if result is None:
                continue
            expiry, strip, weights, weighted_vega = result

            bucket_w = self._BUCKET_WEIGHTS[i]
            n = len(strip)
            legs = []
            legs_data = []
            for j in range(n):
                row = strip.iloc[j]
                qty = (-1.0 * abs(vega_target) * self._BASE_MULT
                       * bucket_w * weights[j] / weighted_vega)
                if abs(qty) < 1e-10:
                    continue
                legs.append(_leg(row, expiry, qty))
                legs_data.append((row, qty))

            if not legs:
                continue

            entry_credit = self._compute_entry_credit(legs_data)
            portfolio.register_new_ptf(symbol, date.normalize(), expiry, legs,
                                        {"mode": self._CODE,
                                         "bucket": i + 1,
                                         "bucket_weight": round(bucket_w, 3),
                                         "entry_credit": entry_credit,
                                         "n_legs": len(legs),
                                         "structure": "varswap_ladder"})


class VSS3Strategy(_MWFLadderStrategy):
    """MWF varswap ladder with extended wings (1-20 delta).

    Captures ultra-deep OTM tail options where the variance premium
    is concentrated, while excluding near-ATM options (|delta| > 0.20).
    """
    _CODE = "VSS3"
    _DELTA_MAX = 0.20
    _DELTA_CUTOFF = 0.01


class VXPSStrategy(_BaseStrategy):
    """VIX-implied put selling (base class for VXS3Strategy).

    Uses VIX to compute N-sigma OTM strike for next 3 MWF expirations.
    Strike: SPOT * (1 - VIX/100 * sqrt((1+DTE)/252) * _SIGMA_FACTOR).
    Daily entry. No signal scaling.
    """
    _CODE = "VXPS"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.02
    _FREQUENCY = "daily"
    _SIGNAL_CONFIG = {}
    _SIGMA_FACTOR = 2.5

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return

        vix = self._sig(symbol, date, "vix", np.nan)
        if not np.isfinite(vix) or vix <= 0:
            return

        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        eligible = chain[(chain["dte"] >= 1) & (chain["dte"] <= 7)]
        if eligible.empty:
            return

        expirations = sorted(eligible["expiration"].unique())
        mwf_exps = [e for e in expirations
                     if pd.to_datetime(e).weekday() in (0, 2, 4)]
        if not mwf_exps:
            mwf_exps = list(expirations)
        mwf_exps = mwf_exps[:3]

        for exp in mwf_exps:
            exp_dt = pd.to_datetime(exp).normalize()
            dte = max(1, (exp_dt - date.normalize()).days)

            if dte <= 2:
                base_mult = 0.02
            elif dte <= 4:
                base_mult = 0.02
            else:
                base_mult = 0.01

            vix_decimal = vix / 100.0
            strike = spot * (1.0 - vix_decimal * np.sqrt((1.0 + dte) / 252.0) * self._SIGMA_FACTOR)

            sub = chain[(chain["expiration"] == exp) & (chain["type"] == "P")]
            if sub.empty:
                continue

            put_row = _select_strike(sub, "P", strike)
            if put_row is None:
                continue

            put_vega = _row_vega(put_row)
            if put_vega < 1e-12:
                continue

            qty = abs(vega_target) * base_mult / put_vega

            legs = [_leg(put_row, exp_dt, -qty)]
            portfolio.register_new_ptf(symbol, date.normalize(), exp_dt, legs,
                                        {"mode": self._CODE, "structure": "vix_put_sell",
                                         "vix": round(vix, 1),
                                         "target_strike": round(strike, 2),
                                         "actual_strike": float(put_row["strike"]),
                                         "dte": dte})


class VXCSStrategy(_BaseStrategy):
    """VIX-implied call selling.

    Mirror of VXPSStrategy but sells OTM calls instead of puts.
    Strike: SPOT * (1 + VIX/100 * sqrt((1+DTE)/252) * _SIGMA_FACTOR).
    Daily entry. No signal scaling.
    """
    _CODE = "VXCS"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.02
    _FREQUENCY = "daily"
    _SIGNAL_CONFIG = {}
    _SIGMA_FACTOR = 1.5

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return

        vix = self._sig(symbol, date, "vix", np.nan)
        if not np.isfinite(vix) or vix <= 0:
            return

        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        eligible = chain[(chain["dte"] >= 1) & (chain["dte"] <= 7)]
        if eligible.empty:
            return

        expirations = sorted(eligible["expiration"].unique())
        mwf_exps = [e for e in expirations
                     if pd.to_datetime(e).weekday() in (0, 2, 4)]
        if not mwf_exps:
            mwf_exps = list(expirations)
        mwf_exps = mwf_exps[:3]

        for exp in mwf_exps:
            exp_dt = pd.to_datetime(exp).normalize()
            dte = max(1, (exp_dt - date.normalize()).days)

            if dte <= 2:
                base_mult = 0.02
            elif dte <= 4:
                base_mult = 0.02
            else:
                base_mult = 0.01

            vix_decimal = vix / 100.0
            strike = spot * (1.0 + vix_decimal * np.sqrt((1.0 + dte) / 252.0) * self._SIGMA_FACTOR)

            sub = chain[(chain["expiration"] == exp) & (chain["type"] == "C")]
            if sub.empty:
                continue

            # Select nearest call at or above target strike (OTM call selection)
            at_or_above = sub[sub["strike"].astype(float) >= strike]
            if not at_or_above.empty:
                pos = (at_or_above["strike"].astype(float) - strike).abs().values.argmin()
                call_row = at_or_above.iloc[pos]
            else:
                pos = (sub["strike"].astype(float) - strike).abs().values.argmin()
                call_row = sub.iloc[pos]

            mid = float(call_row.get("mid", call_row.get("mark", 0)))
            if mid <= 0:
                bid, ask = float(call_row.get("bid", 0)), float(call_row.get("ask", 0))
                mid = 0.5 * (bid + ask) if bid > 0 and ask > 0 else 0
            if mid <= 0:
                continue

            call_vega = _row_vega(call_row)
            if call_vega < 1e-12:
                continue

            qty = abs(vega_target) * base_mult / call_vega

            legs = [_leg(call_row, exp_dt, -qty)]
            portfolio.register_new_ptf(symbol, date.normalize(), exp_dt, legs,
                                        {"mode": self._CODE, "structure": "vix_call_sell",
                                         "vix": round(vix, 1),
                                         "target_strike": round(strike, 2),
                                         "actual_strike": float(call_row["strike"]),
                                         "dte": dte})


class VXS3Strategy(VXPSStrategy):
    """VIX-implied 3.0-sigma OTM put selling, daily, delta-hedged."""
    _CODE = "VXS3"
    _SIGMA_FACTOR = 3.0


class OTMCStrategy(_BaseStrategy):
    """OTM call selling.

    Daily entry. Sells ALL calls with DTE <= 7 and |delta| <= 0.10.
    Groups by expiry. Total daily vega budget = vega_target * 0.03.
    Run as OMDH (delta-hedged) in the final suite.
    """
    _CODE = "OTMC"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.03
    _FREQUENCY = "daily"
    _SIGNAL_CONFIG = {}

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        calls = chain[(chain["type"] == "C") &
                       (chain["dte"] >= 1) &
                       (chain["dte"] <= 7)].copy()
        if calls.empty or "delta" not in calls.columns:
            return

        calls = calls[calls["delta"].astype(float).abs() <= 0.10]
        if calls.empty:
            return

        valid_rows = []
        for idx, row in calls.iterrows():
            mid = _row_mid(row)
            vega = _row_vega(row)
            if mid > 0 and vega > 1e-12:
                valid_rows.append((idx, row, mid, vega))

        if not valid_rows:
            return

        total_vega = sum(v for _, _, _, v in valid_rows)
        if total_vega < 1e-10:
            return

        budget = abs(vega_target) * self._BASE_MULT

        by_expiry = {}
        for idx, row, mid, vega in valid_rows:
            exp = pd.to_datetime(row["expiration"]).normalize()
            by_expiry.setdefault(exp, []).append((row, vega))

        for exp, rows in by_expiry.items():
            exp_vega = sum(v for _, v in rows)
            if exp_vega < 1e-12:
                continue

            legs = []
            for row, vega in rows:
                qty = budget * (vega / total_vega) / vega
                if qty < 1e-12:
                    continue
                legs.append(_leg(row, exp, -qty))

            if legs:
                portfolio.register_new_ptf(symbol, date.normalize(), exp, legs,
                                            {"mode": self._CODE,
                                             "structure": "otm_call_sell",
                                             "n_contracts": len(legs)})


class SDPSStrategy(_BaseStrategy):
    """Short-dated put spread: bull put spread -0.20D/-0.05D, 7D DTE, MWF.

    Signal-driven sizing using HAR VRP, regime score, RV ratio, and
    vol-of-vol z-score.
    """
    _CODE = "SDPS"
    _PROFIT_TARGET = 0.40
    _BASE_MULT = 0.05
    _FREQUENCY = "mwf"

    _SIGNAL_CONFIG = {
        "har_vrp_zscore":   (_zscore_pos, 0.30),
        "regime_score":      (_regime_pos, 0.30),
        "rv_ratio":          (_rv_ratio_low, 0.20),
        "vol_of_vol_z60":   (_zscore_neg, 0.20),
    }

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        self._check_early_exits(date, symbol, state, market)
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        scale = self._signal_scale(symbol, date, self._SIGNAL_CONFIG)

        result = _build_put_spread(chain, date, spot, -0.20, -0.05, 7,
                                    min_dte=3, max_dte=10)
        if result is None:
            return
        expiry, short_put, long_put = result
        short_vega = _row_vega(short_put)
        if short_vega < 1e-10:
            return

        qty = abs(vega_target) * self._BASE_MULT * scale / short_vega
        legs_data = [(short_put, -qty), (long_put, qty)]
        entry_credit = self._compute_entry_credit(legs_data)

        legs = [_leg(short_put, expiry, -qty), _leg(long_put, expiry, qty)]
        portfolio.register_new_ptf(symbol, date.normalize(), expiry, legs,
                                    {"mode": "SDPS", "scale": round(scale, 2),
                                     "entry_credit": entry_credit,
                                     "profit_target": self._PROFIT_TARGET,
                                     "structure": "put_spread_7d"})


# ================================================================
# HEDGE STRATEGIES
# ================================================================

class THTAStrategy(_BaseStrategy):
    """Theta-neutral tail hedge.

    Buy 2-delta puts, sell 10-delta puts sized for theta neutrality:
    qty_buy * |theta_buy| = qty_sell * |theta_sell|.
    1Y DTE target, 9-month hold. Monthly entry on 3rd Fridays. Delta-hedged.
    Zero net theta bleed, positive crash convexity.
    On first day, backfills positions from Jan 2019.
    """
    _CODE = "THTA"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.05
    _FREQUENCY = "monthly"
    _SIGNAL_CONFIG = {}

    _HOLD_MONTHS = 9
    _TARGET_DTE = 365

    def initialize(self, market):
        super().initialize(market)
        self._backfilled = {}

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        # Backfill on first call
        if not self._backfilled.get(symbol, False):
            self._do_backfill(date, symbol, market, portfolio, vega_target)
            self._backfilled[symbol] = True

        if not _is_entry_day(date, self._FREQUENCY):
            return
        self._enter_theta_neutral(date, symbol, market, portfolio,
                                   vega_target, self._TARGET_DTE, self._HOLD_MONTHS)

    def _enter_theta_neutral(self, date, symbol, market, portfolio,
                              vega_target, target_dte, hold_months,
                              min_dte=None, max_dte=None, exit_date_override=None,
                              meta_extra=None):
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        if min_dte is None:
            min_dte = max(30, target_dte - 90)
        if max_dte is None:
            max_dte = target_dte + 90

        result = _build_theta_neutral_puts(chain, date, spot, target_dte,
                                            buy_delta=-0.02, sell_delta=-0.10,
                                            min_dte=min_dte, max_dte=max_dte)
        if result is None:
            return
        expiry, buy_row, sell_row, sell_qty_ratio = result

        buy_vega = _row_vega(buy_row)
        if buy_vega < 1e-12:
            return

        buy_qty = abs(vega_target) * self._BASE_MULT / buy_vega
        sell_qty = buy_qty * sell_qty_ratio

        if exit_date_override is not None:
            exit_date = exit_date_override
        else:
            exit_date = (date + pd.DateOffset(months=hold_months)).normalize()
        if exit_date > expiry:
            exit_date = expiry

        legs = [
            _leg(buy_row, expiry, buy_qty),       # long 2D put (deep OTM)
            _leg(sell_row, expiry, -sell_qty),     # short 10D put (closer OTM)
        ]

        buy_theta = float(buy_row.get("theta", 0))
        sell_theta = float(sell_row.get("theta", 0))

        meta = {"mode": self._CODE,
                "structure": "theta_neutral_puts",
                "hold_months": hold_months,
                "buy_delta": float(buy_row.get("delta", 0)),
                "sell_delta": float(sell_row.get("delta", 0)),
                "sell_qty_ratio": round(sell_qty_ratio, 4),
                "buy_theta": round(buy_theta, 6),
                "sell_theta": round(sell_theta, 6),
                "option_expiry": str(expiry.date()),
                "actual_dte": (expiry - date.normalize()).days}
        if meta_extra:
            meta.update(meta_extra)
        portfolio.register_new_ptf(symbol, date.normalize(), exit_date, legs, meta)

    def _do_backfill(self, date, symbol, market, portfolio, vega_target):
        """Enter positions that would exist if running since Jan 2019."""
        schedule = _compute_backfill_schedule(
            date, self._FREQUENCY, self._HOLD_MONTHS)
        if not schedule:
            return
        for orig_entry, orig_exit, days_elapsed in schedule:
            remaining_dte = max(60, self._TARGET_DTE - days_elapsed)
            if remaining_dte < 60:
                continue
            self._enter_theta_neutral(
                date, symbol, market, portfolio, vega_target,
                target_dte=remaining_dte, hold_months=self._HOLD_MONTHS,
                exit_date_override=orig_exit,
                meta_extra={"backfill": True,
                            "orig_entry": str(orig_entry.date())})


class DVARStrategy(_BaseStrategy):
    """Down-variance swap.

    Long puts at 90-55% moneyness, weighted by 2*dK/K^2. Targets expiry
    beyond 12 months so the strip still has time value at exit.
    Entry on 3rd Fridays. Hold for 12 months. Delta-hedged.
    On first day, backfills positions from Jan 2019.
    """
    _CODE = "DVAR"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.08
    _FREQUENCY = "monthly"
    _SIGNAL_CONFIG = {}

    _HOLD_MONTHS = 12
    _TARGET_DTE = 400

    def initialize(self, market):
        super().initialize(market)
        self._backfilled = {}

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not self._backfilled.get(symbol, False):
            self._do_backfill(date, symbol, market, portfolio, vega_target)
            self._backfilled[symbol] = True

        if not _is_entry_day(date, self._FREQUENCY):
            return
        self._enter_dvar(date, symbol, market, portfolio, vega_target,
                          self._TARGET_DTE, self._HOLD_MONTHS)

    def _enter_dvar(self, date, symbol, market, portfolio, vega_target,
                     target_dte, hold_months, min_dte=None, max_dte=None,
                     exit_date_override=None, meta_extra=None):
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        if min_dte is None:
            min_dte = max(30, target_dte - 70)
        if max_dte is None:
            max_dte = target_dte + 100

        result = _build_down_varswap_strip(
            chain, date, spot, target_dte,
            moneyness_levels=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            min_dte=min_dte, max_dte=max_dte,
        )
        if result is None:
            return
        expiry, rows, weights, weighted_vega = result

        if exit_date_override is not None:
            exit_date = exit_date_override
        else:
            exit_date = (date + pd.DateOffset(months=hold_months)).normalize()
        if exit_date > expiry:
            exit_date = expiry

        legs = []
        for i, row in enumerate(rows):
            qty = abs(vega_target) * self._BASE_MULT * weights[i] / weighted_vega
            if abs(qty) < 1e-10:
                continue
            legs.append(_leg(row, expiry, qty))

        if not legs:
            return

        meta = {"mode": "DVAR", "structure": "down_varswap",
                "n_legs": len(legs), "hold_months": hold_months,
                "option_expiry": str(expiry.date()),
                "target_dte": target_dte,
                "actual_dte": (expiry - date.normalize()).days}
        if meta_extra:
            meta.update(meta_extra)
        portfolio.register_new_ptf(symbol, date.normalize(), exit_date, legs, meta)

    def _do_backfill(self, date, symbol, market, portfolio, vega_target):
        """Enter positions that would exist if running since Jan 2019."""
        schedule = _compute_backfill_schedule(
            date, self._FREQUENCY, self._HOLD_MONTHS)
        for orig_entry, orig_exit, days_elapsed in schedule:
            remaining_dte = max(60, self._TARGET_DTE - days_elapsed)
            if remaining_dte < 60:
                continue
            self._enter_dvar(
                date, symbol, market, portfolio, vega_target,
                target_dte=remaining_dte, hold_months=self._HOLD_MONTHS,
                exit_date_override=orig_exit,
                meta_extra={"backfill": True,
                            "orig_entry": str(orig_entry.date())})


class DVASStrategy(DVARStrategy):
    """Down-variance swap with skew-adjusted delta hedge, 2x sizing.

    Identical positions to DVAR (same put strip, same 2*dK/K^2 weights).
    Delta hedge uses (BS_delta + vega * skew_gradient) instead of BS_delta
    alone. The skew gradient is recomputed daily from live IVs of all legs.
    Over-hedges to capture positive drift of underlying.
    """
    _CODE = "DVAS"
    _BASE_MULT = 0.16

    def adjust_hedge_delta(self, live_ptfs, chain_idx, spot, bs_delta_total):
        """Compute daily skew delta correction across all live portfolios.

        For each portfolio, compute the skew gradient from live IVs:
            skew_grad = (IV_deepOTM - IV_nearATM) / (m_deepOTM - m_nearATM)
        Then for each leg:
            skew_delta_leg = vega * skew_grad
        Sum qty * skew_delta_leg across all legs and portfolios.
        """
        if chain_idx is None or spot is None or spot <= 0:
            return 0.0

        total_skew_delta = 0.0

        for ptf in live_ptfs:
            # Collect (moneyness, iv) and (qty, vega) for legs with live data
            mny_iv = []
            leg_greeks = []  # (qty, vega, strike)
            for leg in ptf.legs:
                if abs(leg.qty) < 1e-12:
                    continue
                if leg.contract_id not in chain_idx.index:
                    continue
                row = chain_idx.loc[leg.contract_id]
                iv_val = float(row.get("implied_volatility", np.nan))
                vega_val = float(row.get("vega", 0.0))
                if np.isfinite(iv_val) and iv_val > 0:
                    mny_iv.append((leg.strike / spot, iv_val))
                    leg_greeks.append((leg.qty, vega_val, leg.strike))

            if len(mny_iv) < 2:
                continue

            # Compute portfolio-level skew gradient from extremes
            mny_iv.sort(key=lambda x: x[0])
            m_lo, iv_lo = mny_iv[0]    # deepest OTM
            m_hi, iv_hi = mny_iv[-1]   # nearest ATM
            if (m_hi - m_lo) < 0.01:
                continue
            # dσ/dm from deep OTM to near ATM: positive for normal put skew
            # (IV_deepOTM > IV_nearATM, m_deepOTM < m_nearATM → neg/neg = pos)
            skew_grad = (iv_lo - iv_hi) / (m_lo - m_hi)

            # Convert to price space:
            #   m = K/S, dm/dS = -K/S²
            #   dσ/dS = dσ/dm × (-K/S²)
            # Natural vega = vega × 100 (vega is per 1% IV = per 0.01σ)
            # Skew delta per option = natural_vega × dσ/dS
            #                       = vega × 100 × dσ/dm × (-K/S²)
            #                       = vega × skew_grad × (-100 × K / S²)
            S2 = spot * spot
            for qty, vega_val, K in leg_greeks:
                total_skew_delta += qty * vega_val * skew_grad * (-100.0 * K / S2)

        return total_skew_delta


class THTA2Strategy(THTAStrategy):
    """Theta-neutral tail hedge v2.

    Buy 2-delta puts, sell 10-delta puts sized for theta neutrality (no ratio cap).
    Targets the 4th monthly 3rd-Friday expiry. 55-day hold. Delta-hedged.
    Monthly entry on 3rd Fridays. Backfills from Jan 2019.
    """
    _CODE = "THT2"
    _HOLD_DAYS = 55

    def _enter_theta_neutral(self, date, symbol, market, portfolio,
                              vega_target, target_dte, hold_months,
                              min_dte=None, max_dte=None, exit_date_override=None,
                              meta_extra=None):
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        # Target the 4th monthly 3rd-Friday expiry
        target_exp = _compute_nth_third_friday(date, n=4)
        if target_exp is None:
            return
        # Match to closest available chain expiry
        expiry = _get_expiry_nearest_date(chain, target_exp)
        if expiry is None:
            return

        result = _build_theta_neutral_puts(chain, date, spot,
                                            buy_delta=-0.02, sell_delta=-0.10,
                                            max_ratio=None, expiry_override=expiry)
        if result is None:
            return
        expiry, buy_row, sell_row, sell_qty_ratio = result

        buy_vega = _row_vega(buy_row)
        if buy_vega < 1e-12:
            return

        buy_qty = abs(vega_target) * self._BASE_MULT / buy_vega
        sell_qty = buy_qty * sell_qty_ratio

        if exit_date_override is not None:
            exit_date = exit_date_override
        else:
            exit_date = (date + pd.Timedelta(days=self._HOLD_DAYS)).normalize()
        if exit_date > expiry:
            exit_date = expiry

        legs = [
            _leg(buy_row, expiry, buy_qty),
            _leg(sell_row, expiry, -sell_qty),
        ]

        buy_theta = float(buy_row.get("theta", 0))
        sell_theta = float(sell_row.get("theta", 0))

        meta = {"mode": self._CODE,
                "structure": "theta_neutral_puts",
                "hold_days": self._HOLD_DAYS,
                "buy_delta": float(buy_row.get("delta", 0)),
                "sell_delta": float(sell_row.get("delta", 0)),
                "sell_qty_ratio": round(sell_qty_ratio, 4),
                "buy_theta": round(buy_theta, 6),
                "sell_theta": round(sell_theta, 6),
                "option_expiry": str(expiry.date()),
                "actual_dte": (expiry - date.normalize()).days}
        if meta_extra:
            meta.update(meta_extra)
        portfolio.register_new_ptf(symbol, date.normalize(), exit_date, legs, meta)

    def _do_backfill(self, date, symbol, market, portfolio, vega_target):
        schedule = _compute_backfill_schedule(
            date, self._FREQUENCY, hold_months=2)  # ~55 days ≈ 2 months
        for orig_entry, orig_exit, days_elapsed in schedule:
            remaining_hold = max(1, self._HOLD_DAYS - days_elapsed)
            exit_dt = (date + pd.Timedelta(days=remaining_hold)).normalize()
            self._enter_theta_neutral(
                date, symbol, market, portfolio, vega_target,
                target_dte=0, hold_months=0,
                exit_date_override=exit_dt,
                meta_extra={"backfill": True,
                            "orig_entry": str(orig_entry.date())})


class XHGEStrategy(_BaseStrategy):
    """Cross-asset crisis detector with macro-triggered long gamma.

    Need >= 2 of 4 triggers: HY OAS z > 1, OVX z > 1, gold mom > 2%,
    VIX term structure inverted > 1.03. Ultra-rare entry (~5-10x/year).
    7D cooldown. Long ATM straddle, delta-hedged.
    """
    _CODE = "XHGE"
    _PROFIT_TARGET = None
    _BASE_MULT = 0.15
    _FREQUENCY = "daily"

    _SIGNAL_CONFIG = {
        "hy_oas_zscore":     (_zscore_pos, 0.30),
        "ovx_z":             (_zscore_pos, 0.25),
        "gld_mom_20d":       (lambda x: np.clip(x * 10.0, -1.0, 1.0), 0.20),
        "vix_ts_ratio":      (lambda x: np.clip((x - 1.0) * 8.0, -1.0, 1.0), 0.25),
    }

    def initialize(self, market):
        super().initialize(market)
        self._last_entry = {}

    def on_day(self, date, symbol, state, market, portfolio, vega_target):
        if not _is_entry_day(date, self._FREQUENCY):
            return
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return

        last = self._last_entry.get(symbol)
        if last is not None and (date - last).days < 7:
            return

        triggers = 0
        if self._sig(symbol, date, "hy_oas_zscore", 0) > 1.0:
            triggers += 1
        if self._sig(symbol, date, "ovx_z", 0) > 1.0:
            triggers += 1
        if self._sig(symbol, date, "gld_mom_20d", 0) > 0.02:
            triggers += 1
        if self._sig(symbol, date, "vix_ts_ratio", 1.0) > 1.03:
            triggers += 1

        if triggers < 2:
            return

        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        scale = self._signal_scale(symbol, date, self._SIGNAL_CONFIG)

        result = _build_straddle(chain, date, spot, 7, min_dte=3, max_dte=10)
        if result is None:
            return
        expiry, call, put = result
        total_vega = _row_vega(call) + _row_vega(put)
        if total_vega < 1e-10:
            return

        qty = abs(vega_target) * self._BASE_MULT * scale / total_vega
        legs = [_leg(call, expiry, qty), _leg(put, expiry, qty)]
        portfolio.register_new_ptf(symbol, date.normalize(), expiry, legs,
                                    {"mode": "XHGE", "scale": round(scale, 2),
                                     "triggers": triggers,
                                     "structure": "straddle_long"})
        self._last_entry[symbol] = date


class TH2LStrategy(THTA2Strategy):
    """Theta-neutral tail hedge with wider strikes and 2x long leg.

    Buy 5-delta puts, sell 15-delta puts sized for theta neutrality.
    Long leg qty is doubled for extra crash convexity.
    4th monthly 3rd-Friday expiry, 55-day hold, fully delta-hedged.
    _BASE_MULT is 3x the base THT2 value (0.15 vs 0.05).
    """
    _CODE = "TH2L"
    _BASE_MULT = 0.15          # 3x THT2's 0.05

    def _enter_theta_neutral(self, date, symbol, market, portfolio,
                              vega_target, target_dte, hold_months,
                              min_dte=None, max_dte=None, exit_date_override=None,
                              meta_extra=None):
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        target_exp = _compute_nth_third_friday(date, n=4)
        if target_exp is None:
            return
        expiry = _get_expiry_nearest_date(chain, target_exp)
        if expiry is None:
            return

        result = _build_theta_neutral_puts(chain, date, spot,
                                            buy_delta=-0.05, sell_delta=-0.15,
                                            max_ratio=None, expiry_override=expiry)
        if result is None:
            return
        expiry, buy_row, sell_row, sell_qty_ratio = result

        buy_vega = _row_vega(buy_row)
        if buy_vega < 1e-12:
            return

        buy_qty = abs(vega_target) * self._BASE_MULT / buy_vega
        buy_qty *= 2.0                              # 2x long leg for convexity
        sell_qty = (buy_qty / 2.0) * sell_qty_ratio  # sell sized off original buy_qty

        if exit_date_override is not None:
            exit_date = exit_date_override
        else:
            exit_date = (date + pd.Timedelta(days=self._HOLD_DAYS)).normalize()
        if exit_date > expiry:
            exit_date = expiry

        legs = [
            _leg(buy_row, expiry, buy_qty),
            _leg(sell_row, expiry, -sell_qty),
        ]

        buy_theta = float(buy_row.get("theta", 0))
        sell_theta = float(sell_row.get("theta", 0))

        meta = {"mode": self._CODE,
                "structure": "theta_neutral_puts_2x_long",
                "hold_days": self._HOLD_DAYS,
                "buy_delta": float(buy_row.get("delta", 0)),
                "sell_delta": float(sell_row.get("delta", 0)),
                "sell_qty_ratio": round(sell_qty_ratio, 4),
                "buy_theta": round(buy_theta, 6),
                "sell_theta": round(sell_theta, 6),
                "long_mult": 2.0,
                "option_expiry": str(expiry.date()),
                "actual_dte": (expiry - date.normalize()).days}
        if meta_extra:
            meta.update(meta_extra)
        portfolio.register_new_ptf(symbol, date.normalize(), exit_date, legs, meta)


class DVMXStrategy(DVASStrategy):
    """Down-variance swap with hybrid max(skew, 20% overhedge) delta hedge.

    Uses finer 2.5% moneyness grid (60-85%, 11 levels) instead of 5% grid.
    Delta hedge: daily max of skew-adjusted correction and 20% overhedge floor,
    whichever produces more overhedge on a given day.
    """
    _CODE = "DVMX"

    _MONEYNESS = [0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75,
                  0.775, 0.80, 0.825, 0.85]

    def _enter_dvar(self, date, symbol, market, portfolio, vega_target,
                     target_dte, hold_months, min_dte=None, max_dte=None,
                     exit_date_override=None, meta_extra=None):
        spot = market.get_spot(symbol, date)
        if spot is None or spot <= 0:
            return
        chain = market.get_chain(symbol, date)
        chain = _ensure_chain(chain, date)
        if chain is None:
            return

        if min_dte is None:
            min_dte = max(30, target_dte - 70)
        if max_dte is None:
            max_dte = target_dte + 100

        result = _build_down_varswap_strip(
            chain, date, spot, target_dte,
            moneyness_levels=self._MONEYNESS,
            min_dte=min_dte, max_dte=max_dte,
        )
        if result is None:
            return
        expiry, rows, weights, weighted_vega = result

        if exit_date_override is not None:
            exit_date = exit_date_override
        else:
            exit_date = (date + pd.DateOffset(months=hold_months)).normalize()
        if exit_date > expiry:
            exit_date = expiry

        legs = []
        for i, row in enumerate(rows):
            qty = abs(vega_target) * self._BASE_MULT * weights[i] / weighted_vega
            if abs(qty) < 1e-10:
                continue
            legs.append(_leg(row, expiry, qty))

        if not legs:
            return

        meta = {"mode": self._CODE, "structure": "down_varswap",
                "n_legs": len(legs), "hold_months": hold_months,
                "option_expiry": str(expiry.date()),
                "target_dte": target_dte,
                "actual_dte": (expiry - date.normalize()).days}
        if meta_extra:
            meta.update(meta_extra)
        portfolio.register_new_ptf(symbol, date.normalize(), exit_date, legs, meta)

    def adjust_hedge_delta(self, live_ptfs, chain_idx, spot, bs_delta_total):
        """Max of skew-adjusted correction and 20% overhedge floor."""
        skew_delta = super().adjust_hedge_delta(live_ptfs, chain_idx, spot,
                                                 bs_delta_total)
        overhedge_floor = -0.2 * bs_delta_total
        # Both positive → more positive = more overhedge (engine negates)
        return max(skew_delta, overhedge_floor)


# ================================================================
# STRATEGY REGISTRY
# ================================================================

_CARRY_CLASSES = [VSS3Strategy, VXS3Strategy, VXCSStrategy, OTMCStrategy, SDPSStrategy]
_HEDGE_CLASSES = [TH2LStrategy, DVMXStrategy, XHGEStrategy]
_ALL_CLASSES = _CARRY_CLASSES + _HEDGE_CLASSES

STRATEGY_REGISTRY_FINAL = {}

for cls in _ALL_CLASSES:
    mode = f"fs_{cls._CODE.lower()}"
    STRATEGY_REGISTRY[mode] = cls
    STRATEGY_REGISTRY_FINAL[cls._CODE] = cls


# ================================================================
# STRATEGY SUITE DEFINITIONS (8 configs)
# ================================================================

OUTPUT_DIR = Path("outputs/flagship_final")


def _make_config(code, cls, delta_hedge=False, symbol="QQQ", max_dte=45, reinvest=True,
                  base_vega_target=0.5, execution_mode="mid_spread"):
    mode = f"fs_{cls._CODE.lower()}"
    cfg = BacktestConfig(
        symbols=[symbol],
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2025-11-16"),
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
        options_cache_dir="cache_options",
        output_path=str(OUTPUT_DIR / f"FS_{code}_{symbol}.xlsx"),
        pnl_explain=True,
        exit_fallback_mode="intrinsic",
        trade_log_mode="light",
    )
    cat = "carry" if cls in _CARRY_CLASSES else "hedge"
    return {"name": f"FS_{code}", "config": cfg, "code": code, "category": cat}


def _build_strategies(symbol="QQQ"):
    return [
        # Carry (5 strategies)
        _make_config("VSS3", VSS3Strategy, delta_hedge=True, symbol=symbol, max_dte=14, base_vega_target=1.5),
        _make_config("VXS3", VXS3Strategy, delta_hedge=True, symbol=symbol, base_vega_target=0.25),
        _make_config("VCBA", VXCSStrategy, delta_hedge=True, symbol=symbol, base_vega_target=0.25,
                     execution_mode="bid_ask"),
        _make_config("OMDH", OTMCStrategy, delta_hedge=True, symbol=symbol, base_vega_target=1.5),
        _make_config("SDPS", SDPSStrategy, delta_hedge=False, symbol=symbol, base_vega_target=1.5),
        # Hedge (3 strategies, reinvest=False -- constant sizing)
        _make_config("TH2L", TH2LStrategy, delta_hedge=True, symbol=symbol, max_dte=500, reinvest=False),
        _make_config("DVMX", DVMXStrategy, delta_hedge=True, symbol=symbol, max_dte=500, reinvest=False),
        _make_config("XHGE", XHGEStrategy, delta_hedge=True, symbol=symbol, reinvest=False),
    ]

STRATEGIES = _build_strategies("QQQ")


# ================================================================
# RUNNER
# ================================================================

def run_suite(strat_filter=None, symbol="QQQ"):
    """Run the flagship final suite."""
    global STRATEGIES
    if symbol != "QQQ":
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
    print(f"FLAGSHIP STRATEGY SUITE -- {len(to_run)} configs on {symbol}")
    print(f"{'=' * 60}")

    # Load raw data directly
    max_dte_needed = max(s["config"].max_dte_for_entry for s in to_run)
    max_rolling_dte = max(s["config"].rolling_max_dte for s in to_run)
    load_cfg = dataclasses.replace(
        to_run[0]["config"],
        max_dte_for_entry=max_dte_needed,
        rolling_max_dte=max_rolling_dte,
        optimized_options_loading=False,
    )

    print(f"\nLoading {symbol} raw data (max_dte={max_dte_needed})...")
    shared_store = _UnifiedDataStore(load_cfg)
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
    parser = argparse.ArgumentParser(description="Flagship Strategy Suite")
    parser.add_argument("--carry-only", action="store_true")
    parser.add_argument("--hedge-only", action="store_true")
    parser.add_argument("--strat", type=str, help="Run single strategy by code (or comma-separated)")
    parser.add_argument("--symbol", type=str, default="QQQ", help="Underlying symbol (default: QQQ)")
    args = parser.parse_args()

    sym = args.symbol.upper()
    if sym != "QQQ":
        STRATEGIES = _build_strategies(sym)

    if args.strat:
        filt = [s.strip().upper() for s in args.strat.split(",")]
    elif args.carry_only:
        filt = [s["code"] for s in STRATEGIES if s["category"] == "carry"]
    elif args.hedge_only:
        filt = [s["code"] for s in STRATEGIES if s["category"] == "hedge"]
    else:
        filt = None

    run_suite(filt, symbol=sym)
