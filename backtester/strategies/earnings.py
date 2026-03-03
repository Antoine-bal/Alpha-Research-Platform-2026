import datetime as dt
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class EarningsStrategy(Strategy):
    """
    Straddle / strangle strategy around earnings events.

    On entry date (determined by timing + entry_lag):
      - "straddle": short/long ATM C+P sized to vega_target
      - "strangle": short/long OTM C+P at +/- moneyness offset

    On exit date (entry + exit_lag): flatten all legs.
    """

    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.entry_exit_map: Dict[str, Dict[pd.Timestamp, Dict[str, Any]]] = {}
        self.signals_df: Optional[pd.DataFrame] = None
        self._short_rules: Dict[int, Optional[Dict]] = {}
        self._long_rules: Dict[int, Optional[Dict]] = {}
        self._ls_rules: Dict[int, Optional[Dict]] = {}

    def initialize(self, market) -> None:
        self._build_entry_exit_map(market)
        if self.config.use_signal:
            self._load_signals_and_build_rules()

    # ------------------------------------------------------------------ #
    # Entry/exit map
    # ------------------------------------------------------------------ #
    def _build_entry_exit_map(self, market) -> None:
        cfg = self.config

        default_entry_lag = {"BMO": -1, "AMC": 0, "DURING": 0, "UNKNOWN": 0}
        default_exit_lag = {"BMO": 0, "AMC": 1, "DURING": 1, "UNKNOWN": 1}

        entry_lag_cfg = {**default_entry_lag, **cfg.entry_lag}
        exit_lag_cfg = {**default_exit_lag, **cfg.exit_lag}

        mapping_global: Dict[str, Dict[pd.Timestamp, Dict[str, Any]]] = {}

        for sym in market.symbols:
            cal = market.get_calendar(sym)
            if len(cal) == 0:
                continue

            evs = market.earnings[market.earnings["symbol"] == sym]
            if evs.empty:
                continue

            mapping_sym: Dict[pd.Timestamp, Dict[str, Any]] = {}

            for _, row in evs.iterrows():
                ev_day = row["event_day"]
                timing = row.get("timing", "UNKNOWN")

                anchor_idx = cal.searchsorted(ev_day)
                if anchor_idx >= len(cal):
                    continue

                entry_lag = entry_lag_cfg.get(timing, 0)
                exit_lag = exit_lag_cfg.get(timing, 1)

                entry_idx = anchor_idx + entry_lag
                exit_idx = anchor_idx + exit_lag

                if not (0 <= entry_idx < len(cal)):
                    continue

                entry_date = cal[entry_idx]
                exit_date = cal[exit_idx] if 0 <= exit_idx < len(cal) else None

                mapping_sym[entry_date] = {
                    "event_day": ev_day,
                    "exit_date": exit_date,
                    "timing": timing,
                }

            mapping_global[sym] = mapping_sym

        self.entry_exit_map = mapping_global

    # ------------------------------------------------------------------ #
    # Strategy.on_day implementation
    # ------------------------------------------------------------------ #
    def on_day(self, date, symbol, state, market, portfolio, vega_target) -> None:
        meta = self.entry_exit_map.get(symbol, {}).get(date)
        if meta is None:
            return

        exit_date = meta.get("exit_date")
        if exit_date is None:
            return

        # Signal-adjusted vega
        if self.config.use_signal:
            vega_target = self._compute_signal_vega(date, symbol, vega_target)
            if abs(vega_target) < 1e-12:
                return

        struct = self.config.earnings_structure
        legs = self._compute_earnings_targets(
            date=date,
            symbol=symbol,
            state=state,
            market=market,
            vega_target=vega_target,
            structure=struct,
        )
        if not legs:
            return

        portfolio.register_new_ptf(
            symbol=symbol,
            entry_date=date,
            exit_date=exit_date,
            legs=legs,
            meta={
                "mode": "earnings",
                "event_day": meta.get("event_day"),
            },
        )

    # ------------------------------------------------------------------ #
    # Option selection & sizing
    # ------------------------------------------------------------------ #
    def _compute_earnings_targets(
        self, date, symbol, state, market, vega_target, structure
    ) -> List[Dict[str, Any]]:
        cfg = self.config

        chain = market.get_chain(symbol, date)
        if chain.empty:
            return []

        if "dte" not in chain.columns:
            chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
            chain["dte"] = (chain["expiration"] - date).dt.days

        min_dte = cfg.min_dte_for_entry
        max_dte = cfg.max_dte_for_entry
        eligible = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]
        if eligible.empty:
            return []

        first = eligible.sort_values("expiration").iloc[0]
        expiry = first["expiration"]
        sub = chain[chain["expiration"] == expiry]
        if sub.empty:
            return []

        if ("moneyness" not in sub.columns) or sub["moneyness"].isna().all():
            spot_today = market.get_spot(symbol, date)
            if spot_today is None or not np.isfinite(spot_today):
                return []
            if "strike_eff" in sub.columns:
                sub["moneyness"] = sub["strike_eff"] / spot_today
            else:
                sub["moneyness"] = sub["strike"] / spot_today

        calls = sub[sub["type"] == "C"]
        puts = sub[sub["type"] == "P"]
        if calls.empty or puts.empty:
            return []

        if structure == "strangle":
            offset = cfg.strangle_mny_offset
            calls["mny_diff"] = (calls["moneyness"] - (1.0 + offset)).abs()
            puts["mny_diff"] = (puts["moneyness"] - (1.0 - offset)).abs()
        else:  # straddle
            calls["mny_diff"] = (calls["moneyness"] - 1.0).abs()
            puts["mny_diff"] = (puts["moneyness"] - 1.0).abs()

        call = calls.sort_values("mny_diff").iloc[0]
        put = puts.sort_values("mny_diff").iloc[0]

        return self._size_vega_legs(symbol, [call, put], vega_target)

    def _size_vega_legs(
        self, symbol: str, opt_rows: list, vega_target: float
    ) -> List[Dict[str, Any]]:
        direction = -1.0 if vega_target > 0 else 1.0

        if not opt_rows or abs(vega_target) < 1e-12:
            return []

        mag = abs(vega_target)
        legs_with_vega = []
        for row in opt_rows:
            v = float(row.get("vega", np.nan))
            if not np.isfinite(v) or abs(v) < 1e-12:
                continue
            legs_with_vega.append((row, v))

        if not legs_with_vega:
            return []

        denom = sum(abs(v) for (_, v) in legs_with_vega)
        if denom <= 1e-12:
            return []

        scale = mag / denom

        targets: List[Dict[str, Any]] = []
        for row, v_per_contract in legs_with_vega:
            qty = direction * scale
            targets.append({
                "contract_id": str(row["contractID"]),
                "symbol": symbol,
                "expiry": pd.to_datetime(row["expiration"]).normalize(),
                "strike": float(row["strike"]),
                "type": str(row["type"]).upper(),
                "qty": qty,
            })

        return targets

    # ------------------------------------------------------------------ #
    # Signal routing
    # ------------------------------------------------------------------ #
    def _load_signals_and_build_rules(self):
        cfg = self.config
        path = cfg.signal_csv_path
        if not os.path.exists(path):
            return
        d = pd.read_csv(path, parse_dates=["EventDate", "AnchorDate"])

        if "MLShortZ" in d.columns and "MLLongZ" in d.columns:
            d["ShortScore_z"] = d["MLShortZ"]
            d["LongScore_z"] = d["MLLongZ"]
        if "RealizedPnL" in d.columns:
            d["PnL_proxy"] = d["RealizedPnL"]

        d = d.dropna(subset=["ShortScore_z", "LongScore_z", "PnL_proxy"])
        d["year"] = d["AnchorDate"].dt.year
        self.signals_df = d

        min_years = cfg.signal_min_years
        n_bins = cfg.signal_n_bins
        max_mult = cfg.signal_max_vega_mult

        def quant_edges(s):
            qs = np.linspace(0, 1, n_bins + 1)
            e = np.quantile(s.dropna(), qs)
            e = np.unique(e)
            return e if len(e) > 1 else None

        def train_rule(train_df, score_col, pnl_col, flip=False):
            t = train_df[[score_col, pnl_col]].dropna()
            if t.empty:
                return None
            s = t[score_col]
            edges = quant_edges(s)
            if edges is None:
                return None
            bins = np.searchsorted(edges, s.values, side="right") - 1
            pnl = t[pnl_col].values
            if flip:
                pnl = -pnl
            g = pd.Series(pnl).groupby(bins).mean()
            g = g[g.index >= 0]
            good = g[g > 0.0]
            if good.empty:
                return None
            mm = good.clip(lower=0.0)
            vmax = mm.max()
            mult = (mm / vmax * max_mult) if vmax > 0 else mm * 0.0
            vega_mult = {int(b): float(v) for b, v in mult.items()}
            return {
                "edges": edges,
                "good_bins": set(int(b) for b in good.index),
                "vega_mult": vega_mult,
            }

        years = sorted(d["year"].unique())
        for y in years:
            y_start = dt.datetime(y, 1, 1)
            train_start = y_start - dt.timedelta(days=365 * min_years)
            train = d[(d["AnchorDate"] >= train_start) & (d["AnchorDate"] < y_start)]
            if train["year"].nunique() < min_years or len(train) < 50:
                self._short_rules[y] = None
                self._long_rules[y] = None
                self._ls_rules[y] = None
                continue
            short_rule = train_rule(train, "ShortScore_z", "PnL_proxy", flip=False)
            long_rule = train_rule(train, "LongScore_z", "PnL_proxy", flip=True)
            self._short_rules[y] = short_rule
            self._long_rules[y] = long_rule
            ls_rule = None
            if short_rule and long_rule:
                ls_rule = {
                    "short_edges": short_rule["edges"],
                    "long_edges": long_rule["edges"],
                    "short_bins": short_rule["good_bins"],
                    "long_bins": long_rule["good_bins"],
                    "short_mult": short_rule["vega_mult"],
                    "long_mult": long_rule["vega_mult"],
                }
            self._ls_rules[y] = ls_rule

    def _assign_bin(self, x, edges):
        if edges is None or pd.isna(x):
            return None
        return int(np.searchsorted(edges, x, side="right") - 1)

    def _signal_decision_for_row(
        self, row: pd.Series, mode: str
    ) -> Tuple[str, float]:
        y = int(row["AnchorDate"].year)

        if mode in ("ls", "long_short", "long-short"):
            mode = "long_short"

        if mode == "short":
            rule = self._short_rules.get(y)
            if not rule:
                return "flat", 0.0
            s = float(row["ShortScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]:
                return "flat", 0.0
            v = rule["vega_mult"].get(b, 0.0)
            return ("short", v if v > 0 else 0.0)

        if mode == "long":
            rule = self._long_rules.get(y)
            if not rule:
                return "flat", 0.0
            s = float(row["LongScore_z"])
            b = self._assign_bin(s, rule["edges"])
            if b is None or b not in rule["good_bins"]:
                return "flat", 0.0
            v = rule["vega_mult"].get(b, 0.0)
            return ("long", v if v > 0 else 0.0)

        if mode == "long_short":
            rule = self._ls_rules.get(y)
            if not rule:
                return "flat", 0.0

            s_s = float(row["ShortScore_z"])
            s_l = float(row["LongScore_z"])
            b_s = self._assign_bin(s_s, rule["short_edges"])
            b_l = self._assign_bin(s_l, rule["long_edges"])

            candidates = []
            if b_s is not None and b_s in rule["short_bins"]:
                v_s = rule["short_mult"].get(b_s, 0.0)
                if v_s > 0:
                    candidates.append(("short", v_s))
            if b_l is not None and b_l in rule["long_bins"]:
                v_l = rule["long_mult"].get(b_l, 0.0)
                if v_l > 0:
                    candidates.append(("long", v_l))

            if not candidates:
                return "flat", 0.0

            candidates.sort(key=lambda x: x[1], reverse=True)
            if len(candidates) >= 2 and candidates[0][1] == candidates[1][1]:
                for side, v in candidates:
                    if side == "short":
                        return "short", v
            return candidates[0]

        return "flat", 0.0

    def _compute_signal_vega(
        self, date: pd.Timestamp, symbol: str, base_vega: float
    ) -> float:
        if self.signals_df is None:
            return base_vega

        meta = self.entry_exit_map.get(symbol, {}).get(date)
        if not meta:
            return 0.0

        event_day = pd.to_datetime(meta["event_day"]).normalize()
        d = self.signals_df
        rows = d[(d["Symbol"] == symbol) & (d["EventDate"] == event_day)]
        if rows.empty:
            return 0.0

        row = rows.iloc[0]

        # --- Simple threshold filter (overrides walk-forward bins) ---
        fcol = self.config.signal_filter_col
        if fcol:
            val = float(row.get(fcol, np.nan))
            if np.isnan(val) or val < self.config.signal_filter_min:
                return 0.0
            return base_vega  # full position for events passing the filter

        # --- Walk-forward bin system (original) ---
        mode = self.config.signal_mode
        side, mult = self._signal_decision_for_row(row, mode)
        if side == "flat" or mult <= 0:
            return 0.0

        v = base_vega * mult
        return v if side == "short" else -v
