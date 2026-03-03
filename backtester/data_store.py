import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig


def _apply_option_quality_filters(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    if df.empty:
        return df

    out = df

    min_oi = cfg.min_open_interest
    if min_oi is not None and "open_interest" in out.columns:
        out = out[out["open_interest"].fillna(0).astype(float) >= float(min_oi)]

    min_vol = cfg.min_volume
    if min_vol is not None and "volume" in out.columns:
        out = out[out["volume"].fillna(0).astype(float) >= float(min_vol)]

    max_spread_pct = cfg.max_bid_ask_spread_pct
    if max_spread_pct is not None and ("bid" in out.columns) and ("ask" in out.columns):
        bid = out["bid"].astype(float)
        ask = out["ask"].astype(float)
        mid = out["mid"] if "mid" in out.columns else 0.5 * (bid + ask)
        spread = ask - bid
        spread_pct = np.where(mid > 0, spread / mid, np.nan)
        out = out[
            (spread >= 0)
            & np.isfinite(spread_pct)
            & (spread_pct <= float(max_spread_pct))
        ]

    return out


class DataStore:
    """
    Loads and provides:
      - daily split-normalized spot
      - full options chain per (symbol, date)
      - earnings events + entry/exit mapping
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.symbols = config.symbols
        self.corp_dir = pathlib.Path(config.corp_dir)
        self.options_dir = pathlib.Path(config.options_dir)
        self.start_date = config.start_date
        self.end_date = config.end_date

        self.spot: Dict[str, pd.DataFrame] = {}
        self.options: Dict[str, pd.DataFrame] = {}
        self.earnings: pd.DataFrame = pd.DataFrame()
        self.spot_cache: Dict[str, Dict[pd.Timestamp, float]] = {}
        self.rates: Optional[pd.Series] = None

        self._load_spot()
        self._load_earnings()
        self._load_rates()

    # ---------- Spot ----------
    def _load_spot(self):
        for sym in self.symbols:
            path = self.corp_dir / f"{sym}_daily_adjusted.parquet"
            if not path.exists():
                print(f"[WARN] Spot file missing for {sym}: {path}")
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

            if "close" in df.columns:
                close_raw = df["close"].astype(float)
            elif "adj_close" in df.columns:
                close_raw = df["adj_close"].astype(float)
            else:
                raise ValueError(f"{sym}: neither 'close' nor 'adj_close' in {path}")

            if "split_coeff" in df.columns:
                split_raw = df["split_coeff"].astype(float)
                split_raw = split_raw.replace(0.0, np.nan).fillna(1.0)
            else:
                split_raw = pd.Series(1.0, index=df.index)

            split_level = split_raw.cumprod()
            level_last = float(split_level.iloc[-1])
            if level_last <= 0:
                level_last = 1.0

            price_factor = level_last / split_level
            spot_norm = close_raw / price_factor

            out = pd.DataFrame({
                "spot": close_raw,
                "split_factor": split_raw.astype(float),
            })
            out.index = split_level.index
            self.spot[sym] = out
            print(f"[INFO] Spot loaded for {sym}: {out.shape[0]} days")

    # ---------- Risk-free rate ----------
    def _load_rates(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "risk_free_3m.parquet")
        if not os.path.exists(path):
            # Try relative to CWD
            path = "risk_free_3m.parquet"
        if not os.path.exists(path):
            return
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        self.rates = df.set_index("date")["rf_annual"].sort_index()
        print(f"[INFO] Risk-free rates loaded: {len(self.rates)} days")

    def get_rate(self, date: pd.Timestamp) -> float:
        if self.rates is None:
            return 0.0
        try:
            return float(self.rates.loc[date])
        except KeyError:
            idx = self.rates.index.searchsorted(date)
            if idx > 0:
                return float(self.rates.iloc[idx - 1])
            return 0.0

    def get_spot(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        sym_cache = self.spot_cache.setdefault(symbol, {})
        cached = sym_cache.get(date)
        if cached is not None:
            return cached

        df = self.spot.get(symbol)
        if df is None:
            spot = self._get_spot_from_options(symbol, date)
            if spot is not None:
                sym_cache[date] = spot
            return spot
        try:
            spot = float(df.loc[date, "spot"])
            sym_cache[date] = spot
            return spot
        except KeyError:
            spot = self._get_spot_from_options(symbol, date)
            if spot is not None:
                sym_cache[date] = spot
            return spot

    def get_split_factor(self, symbol: str, date: pd.Timestamp) -> float:
        df = self.spot.get(symbol)
        if df is None:
            return 1.0
        try:
            return float(df.loc[date, "split_factor"])
        except KeyError:
            return 1.0

    def get_spot_calendar(self, symbol: str) -> pd.DatetimeIndex:
        df = self.spot.get(symbol)
        if df is None:
            return pd.DatetimeIndex([])
        return df.index

    def get_calendar(self, symbol: str) -> pd.DatetimeIndex:
        spot_cal = self.get_spot_calendar(symbol)
        if len(spot_cal) > 0:
            return spot_cal
        opt_cal = self._get_option_calendar(symbol)
        return opt_cal if opt_cal is not None else pd.DatetimeIndex([])

    def _get_option_calendar(self, symbol: str) -> Optional[pd.DatetimeIndex]:
        if symbol in self.options and not self.options[symbol].empty:
            return pd.DatetimeIndex(self.options[symbol]["date"].unique())

        path = self.options_dir / f"{symbol}.parquet"
        if not path.exists():
            return None
        try:
            df = pd.read_parquet(path, columns=["date"])
        except Exception:
            return None
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return pd.DatetimeIndex(df["date"].unique())

    def _get_spot_from_options(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        if symbol not in self.options:
            self.load_symbol_options(symbol)

        df = self.options.get(symbol)
        if df is None or df.empty:
            return None

        try:
            sub = df.loc[date]
        except KeyError:
            return None
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        if sub.empty:
            return None

        for col in [
            "underlying_price", "underlying", "underlying_last",
            "underlying_close", "spot",
        ]:
            if col in sub.columns:
                vals = sub[col].astype(float)
                vals = vals[np.isfinite(vals)]
                if not vals.empty:
                    return float(vals.median())

        if "moneyness" in sub.columns:
            mny = sub["moneyness"].astype(float)
            strikes = sub["strike"].astype(float)
            mask = np.isfinite(mny) & np.isfinite(strikes) & (mny > 0)
            if mask.any():
                est = (strikes[mask] / mny[mask]).astype(float)
                est = est[np.isfinite(est)]
                if not est.empty:
                    return float(est.median())

        if "delta" in sub.columns and "type" in sub.columns:
            delta = sub["delta"].astype(float)
            strikes = sub["strike"].astype(float)
            opt_type = sub["type"].astype(str).str.upper().str[0]
            call_mask = (opt_type == "C") & np.isfinite(delta) & np.isfinite(strikes)
            if call_mask.any():
                idx = (delta[call_mask] - 0.5).abs().idxmin()
                val = strikes.loc[idx]
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
            put_mask = (opt_type == "P") & np.isfinite(delta) & np.isfinite(strikes)
            if put_mask.any():
                idx = (delta[put_mask] + 0.5).abs().idxmin()
                val = strikes.loc[idx]
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)
        return None

    def _prepare_option_chain(self, df: pd.DataFrame, sym: str) -> pd.DataFrame:
        spot_df = self.spot.get(sym)
        if spot_df is not None and "split_factor" in spot_df.columns:
            pf = spot_df[["split_factor"]].reset_index()
            df = df.merge(pf, on="date", how="left")
        else:
            df["split_factor"] = 1.0
        df["split_factor"] = df["split_factor"].fillna(1.0).astype(float)

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

        df = df[df["mid"] > 0]

        if "bid" in df.columns:
            df["bid"] = df["bid"].astype(float)
        if "ask" in df.columns:
            df["ask"] = df["ask"].astype(float)

        for greek in ["delta", "gamma", "vega", "theta"]:
            if greek in df.columns:
                df[greek] = df[greek].astype(float)

        df = _apply_option_quality_filters(df, self.config)

        df["strike_eff"] = df["strike"] / df["split_factor"]

        if ("moneyness" not in df.columns) or df["moneyness"].isna().all():
            if spot_df is not None and not spot_df.empty:
                spot_merge = (
                    spot_df[["spot"]]
                    .reset_index()
                    .rename(columns={"spot": "_spot_for_mny"})
                )
                df = df.merge(spot_merge, on="date", how="left")
                df["_spot_for_mny"] = df["_spot_for_mny"].astype(float)
                df["moneyness"] = np.where(
                    df["_spot_for_mny"] > 0,
                    df["strike_eff"] / df["_spot_for_mny"],
                    np.nan,
                )
                df.drop(columns=["_spot_for_mny"], inplace=True)
            elif "underlying_price" in df.columns:
                up = df["underlying_price"].astype(float)
                df["moneyness"] = np.where(up > 0, df["strike_eff"] / up, np.nan)
            else:
                df["moneyness"] = np.nan

        return df.drop_duplicates()

    # ---------- Options ----------

    # Columns needed by the backtester (everything else is dropped in cache)
    _OPT_COLUMNS = [
        "date", "expiration", "strike", "type", "contractID",
        "bid", "ask", "mark", "last",
        "delta", "gamma", "vega", "theta", "implied_volatility",
        "open_interest", "volume",
        "dte",  # pre-computed in cache for predicate pushdown
    ]

    def load_symbol_options(self, sym: str) -> None:
        if sym in self.options:
            return

        if self.config.optimized_options_loading:
            self._load_options_optimized(sym)
        else:
            self._load_options_legacy(sym)

    def _load_options_legacy(self, sym: str) -> None:
        """Original full-chain loading path (no column pruning, no cache)."""
        path = self.options_dir / f"{sym}.parquet"
        if not path.exists():
            print(f"[WARN] Options file missing for {sym}: {path}")
            return

        df = pd.read_parquet(path)
        self._ingest_options_df(df, sym)

    _CACHE_VERSION = "v4"  # bump to force cache rebuild with new filters

    def _load_options_optimized(self, sym: str) -> None:
        """Optimized path: use date-sorted, column-pruned cache with predicate pushdown."""
        cache_dir = pathlib.Path(self.config.options_cache_dir)
        cache_path = cache_dir / f"{sym}_{self._CACHE_VERSION}.parquet"
        raw_path = self.options_dir / f"{sym}.parquet"

        if not raw_path.exists():
            print(f"[WARN] Options file missing for {sym}: {raw_path}")
            return

        # Lazy cache build
        if not cache_path.exists():
            self._build_optimized_cache(sym, raw_path, cache_path)

        # Read with predicate pushdown on date AND DTE
        import pyarrow.parquet as pq
        max_dte_cutoff = max(
            self.config.max_dte_for_entry + 30,
            self.config.rolling_max_dte + 30,
        )
        filters = [
            ("date", ">=", self.start_date),
            ("date", "<=", self.end_date),
            ("dte", ">=", 0),
            ("dte", "<=", max_dte_cutoff),
        ]
        # Only request columns that exist in the cached file
        pf = pq.ParquetFile(cache_path)
        available = set(pf.schema_arrow.names)
        read_cols = [c for c in self._OPT_COLUMNS if c in available]

        df = pd.read_parquet(cache_path, columns=read_cols, filters=filters)
        self._ingest_options_df(df, sym)

    def _build_optimized_cache(
        self,
        sym: str,
        raw_path: pathlib.Path,
        cache_path: pathlib.Path,
        max_dte_cache: int = 45,
        min_mny: float = 0.70,
        max_mny: float = 1.30,
    ) -> None:
        """Build a date-sorted, column-pruned, DTE+moneyness-filtered cache.

        Uses chunked pandas processing to handle large raw files without
        loading the entire dataset into memory at once.
        Filters by DTE (<=45) AND moneyness (0.70-1.30) using spot data.
        """
        import gc
        import pyarrow as pa
        import pyarrow.parquet as pq

        print(f"[CACHE] Building {self._CACHE_VERSION} cache for {sym} "
              f"(DTE<={max_dte_cache}, mny=[{min_mny}-{max_mny}])...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Build spot lookup for moneyness filtering
        spot_df = self.spot.get(sym)
        spot_lookup = {}
        if spot_df is not None and not spot_df.empty:
            spot_lookup = spot_df["spot"].to_dict()
            print(f"[CACHE] Spot lookup: {len(spot_lookup)} dates for moneyness filter")

        # Determine available columns
        pf = pq.ParquetFile(raw_path)
        available = set(pf.schema_arrow.names)
        base_cols = [c for c in DataStore._OPT_COLUMNS if c in available and c != "dte"]
        raw_rows = pf.metadata.num_rows
        print(f"[CACHE] {sym}: {raw_rows:,} raw rows, processing in chunks...")

        # Phase 1: Read in chunks, filter by DTE + moneyness, write to temp
        tmp_path = cache_path.with_suffix(".tmp.parquet")
        writer = None
        filtered_rows = 0

        for batch in pf.iter_batches(batch_size=500_000, columns=base_cols):
            chunk_df = pa.Table.from_batches([batch]).to_pandas()

            # Compute DTE
            chunk_df["date"] = pd.to_datetime(chunk_df["date"])
            chunk_df["expiration"] = pd.to_datetime(chunk_df["expiration"])
            chunk_df["dte"] = (chunk_df["expiration"] - chunk_df["date"]).dt.days

            # Filter DTE
            chunk_df = chunk_df[(chunk_df["dte"] >= 0) & (chunk_df["dte"] <= max_dte_cache)]
            if chunk_df.empty:
                del chunk_df
                continue

            # Filter moneyness using spot lookup
            if spot_lookup:
                spot_vals = chunk_df["date"].dt.normalize().map(spot_lookup)
                valid_spot = spot_vals.notna() & (spot_vals > 0)
                mny = chunk_df["strike"].astype(float) / spot_vals.astype(float)
                # Keep rows where: no spot data OR moneyness in range
                keep = ~valid_spot | ((mny >= min_mny) & (mny <= max_mny))
                chunk_df = chunk_df[keep]
                if chunk_df.empty:
                    del chunk_df
                    continue

            t = pa.Table.from_pandas(chunk_df, preserve_index=False)
            filtered_rows += t.num_rows
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, t.schema)
            writer.write_table(t)
            del chunk_df

        if writer is not None:
            writer.close()
        del writer
        gc.collect()

        mny_str = f" + mny[{min_mny}-{max_mny}]" if spot_lookup else ""
        print(f"[CACHE] {sym}: {raw_rows:,} -> {filtered_rows:,} after DTE<={max_dte_cache}{mny_str}")

        if filtered_rows == 0:
            print(f"[WARN] {sym}: no rows after filters!")
            return

        # Phase 2: Read the (much smaller) filtered file, sort, normalize, write final
        df = pd.read_parquet(tmp_path)

        # Delete temp file FIRST to free disk space before writing final cache
        try:
            tmp_path.unlink()
        except OSError:
            pass

        # Normalize types
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
        df["strike"] = df["strike"].astype(float)
        df["type"] = df["type"].astype(str).str.upper().str[0]
        if "dte" not in df.columns:
            df["dte"] = (df["expiration"] - df["date"]).dt.days

        # Sort by date
        df.sort_values("date", inplace=True, ignore_index=True)

        # Write final cache
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, cache_path, row_group_size=100_000)

        raw_mb = raw_path.stat().st_size / (1024 * 1024)
        cache_mb = cache_path.stat().st_size / (1024 * 1024)
        print(
            f"[CACHE] {sym}: {raw_mb:.0f} MB -> {cache_mb:.0f} MB "
            f"({cache_mb/raw_mb:.0%}), {len(df):,} rows, "
            f"sorted by date with {(len(df) + 99_999) // 100_000} row groups"
        )
        del df, table
        gc.collect()

    def _ingest_options_df(self, df: pd.DataFrame, sym: str) -> None:
        """Common processing after reading options DataFrame (legacy or optimized)."""
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.normalize()
        df["strike"] = df["strike"].astype(float)
        df["type"] = df["type"].astype(str).str.upper().str[0]

        df = df[(df["date"] >= self.start_date) & (df["date"] <= self.end_date)]
        if df.empty:
            print(f"[WARN] No option rows for {sym} in backtest window.")
            return

        df["dte"] = (df["expiration"] - df["date"]).dt.days

        # Early DTE filter to reduce memory before heavy processing
        max_dte_cutoff = max(
            self.config.max_dte_for_entry + 30,
            self.config.rolling_max_dte + 30,
        )
        df = df[(df["dte"] >= 0) & (df["dte"] <= max_dte_cutoff)]
        if df.empty:
            print(f"[WARN] No option rows for {sym} after DTE filter.")
            return

        # Early moneyness filter if possible
        if "strike" in df.columns:
            spot_df = self.spot.get(sym)
            if spot_df is not None and not spot_df.empty:
                spot_merge = spot_df[["spot"]].reset_index().rename(
                    columns={"spot": "_spot_approx"}
                )
                df = df.merge(spot_merge, on="date", how="left")
                approx_mny = df["strike"].astype(float) / df["_spot_approx"].astype(float)
                df = df[
                    (approx_mny >= self.config.min_moneyness)
                    & (approx_mny <= self.config.max_moneyness)
                ]
                df.drop(columns=["_spot_approx"], inplace=True)

        if df.empty:
            print(f"[WARN] No option rows for {sym} after moneyness filter.")
            return

        df = self._prepare_option_chain(df, sym)
        df = df.sort_values("date").set_index("date", drop=False)
        self.options[sym] = df
        print(f"[INFO] Options loaded for {sym}: {df.shape[0]:,} rows")

    def get_chain(self, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
        df = self.options.get(symbol)
        if df is None:
            return pd.DataFrame()
        try:
            sub = df.loc[date]
        except KeyError:
            return pd.DataFrame()
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        return sub.copy()

    # ---------- Earnings ----------
    def _load_earnings(self):
        if not os.path.exists(self.config.earnings_csv):
            print(f"[WARN] Earnings file {self.config.earnings_csv} missing.")
            self.earnings = pd.DataFrame(columns=["symbol", "event_day"])
            return

        df = pd.read_csv(self.config.earnings_csv)
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["event_day"] = pd.to_datetime(df["event_day"]).dt.normalize()
        if "timing" in df.columns:
            df["timing"] = df["timing"].astype(str).str.upper()
        else:
            df["timing"] = "UNKNOWN"
        df = df[df["symbol"].isin(self.symbols)]
        df = df[(df["event_day"] >= self.start_date) & (df["event_day"] <= self.end_date)]
        df = df.drop_duplicates(subset=["symbol", "event_day"]).reset_index(drop=True)

        self.earnings = df
        print(f"[INFO] Earnings loaded: {len(df)} events")
