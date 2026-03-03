"""
Dispersion strategy: trade index vol vs component vol.

Long dispersion (direction=1):
  - Buy component straddles, sell index straddle
  - Profits when realized correlation < implied correlation
  (the "correlation risk premium" trade)

Short dispersion (direction=-1):
  - Sell component straddles, buy index straddle
  - Profits when realized correlation > implied correlation

All symbols (index + components) must be in config.symbols.
The strategy uses on_day() per symbol:
  - For the index: trades in the opposite direction to dispersion_direction
  - For each component: trades in the dispersion_direction, scaled by 1/N
"""

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class DispersionStrategy(Strategy):
    """Dispersion trading: index vol vs component vol."""

    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self._index_sym: str = config.dispersion_index_symbol.upper()
        self._component_syms: Set[str] = set(
            s.upper() for s in config.dispersion_component_symbols
        )
        self._n_components: int = max(1, len(self._component_syms))

    # ------------------------------------------------------------------
    # Entry frequency
    # ------------------------------------------------------------------
    def _is_entry_day(self, date: pd.Timestamp) -> bool:
        freq = self.config.dispersion_entry_frequency
        if freq == "daily":
            return True
        elif freq == "weekly":
            return date.weekday() == self.config.dispersion_entry_weekday
        elif freq == "monthly_opex":
            return self._is_third_friday(date)
        return True

    @staticmethod
    def _is_third_friday(date: pd.Timestamp) -> bool:
        if date.weekday() != 4:
            return False
        return 15 <= date.day <= 21

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def on_day(self, date, symbol, state, market, portfolio, vega_target) -> None:
        cfg = self.config

        if not self._is_entry_day(date):
            return

        if vega_target is None or abs(vega_target) < 1e-12:
            return

        # Determine role
        sym_upper = symbol.upper()
        is_index = sym_upper == self._index_sym
        is_component = sym_upper in self._component_syms

        if not is_index and not is_component:
            return

        # Direction logic
        # dispersion_direction=1 → long dispersion: long component vol, short index vol
        disp_dir = cfg.dispersion_direction
        if is_index:
            trade_direction = -disp_dir   # opposite to dispersion direction
        else:
            trade_direction = disp_dir    # same as dispersion direction
            # Scale component vega so total component vega ~ index vega
            vega_target = vega_target / self._n_components

        chain = market.get_chain(symbol, date)
        if chain is None or chain.empty:
            return

        # Normalize columns
        if "expiration" not in chain.columns:
            if "expiry" in chain.columns:
                chain = chain.rename(columns={"expiry": "expiration"})
            else:
                return

        if "dte" not in chain.columns:
            chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
            chain["dte"] = (chain["expiration"] - date.normalize()).dt.days

        spot = market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return

        # --- Select expiry ---
        target_dte = cfg.dispersion_target_dte
        min_dte = cfg.min_dte_for_entry
        max_dte = cfg.max_dte_for_entry

        sub_chain = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]
        if sub_chain.empty:
            return

        exp_dte = (
            sub_chain[["expiration", "dte"]]
            .drop_duplicates()
            .groupby("expiration")["dte"]
            .min()
        )
        eligible = exp_dte[exp_dte >= target_dte]
        if eligible.empty:
            eligible = exp_dte[exp_dte >= 1]
        if eligible.empty:
            return
        expiry = eligible.idxmin()
        expiry_date = pd.to_datetime(expiry).normalize()

        sub = sub_chain[sub_chain["expiration"] == expiry]
        if sub.empty:
            return

        # Moneyness for ATM selection
        if "moneyness" not in sub.columns or sub["moneyness"].isna().all():
            sub = sub.copy()
            if "strike_eff" in sub.columns:
                sub["moneyness"] = sub["strike_eff"] / spot
            else:
                sub["moneyness"] = sub["strike"] / spot

        # --- Select ATM straddle / strangle ---
        structure = cfg.dispersion_structure

        calls = sub[sub["type"] == "C"].copy()
        puts = sub[sub["type"] == "P"].copy()
        if calls.empty or puts.empty:
            return

        if structure == "strangle":
            offset = cfg.strangle_mny_offset
            calls["mny_diff"] = (calls["moneyness"] - (1.0 + offset)).abs()
            puts["mny_diff"] = (puts["moneyness"] - (1.0 - offset)).abs()
        else:  # straddle
            calls["mny_diff"] = (calls["moneyness"] - 1.0).abs()
            puts["mny_diff"] = (puts["moneyness"] - 1.0).abs()

        call_row = calls.sort_values("mny_diff").iloc[0]
        put_row = puts.sort_values("mny_diff").iloc[0]

        # Vega sizing
        call_vega = float(call_row.get("vega", 0.0))
        put_vega = float(put_row.get("vega", 0.0))
        total_vega = abs(call_vega) + abs(put_vega)
        if total_vega < 1e-12:
            return

        scale = abs(vega_target) / total_vega
        qty = trade_direction * scale

        if abs(qty) < 1e-10:
            return

        # Build legs
        legs: List[Dict[str, Any]] = [
            {
                "contract_id": str(call_row["contractID"]),
                "expiry": expiry_date,
                "strike": float(call_row["strike"]),
                "type": "C",
                "qty": float(qty),
            },
            {
                "contract_id": str(put_row["contractID"]),
                "expiry": expiry_date,
                "strike": float(put_row["strike"]),
                "type": "P",
                "qty": float(qty),
            },
        ]

        # Exit date
        holding_days = max(1, cfg.dispersion_holding_days)
        bdays = pd.bdate_range(start=date.normalize(), periods=holding_days)
        target_close_date = bdays[-1]
        exit_date = min(expiry_date, target_close_date)

        portfolio.register_new_ptf(
            symbol=symbol,
            entry_date=date.normalize(),
            exit_date=exit_date,
            legs=legs,
            meta={
                "mode": "dispersion",
                "role": "index" if is_index else "component",
                "direction": float(trade_direction),
                "structure": structure,
                "target_dte": target_dte,
                "expiry_dte": int(exp_dte.loc[expiry]),
            },
        )
