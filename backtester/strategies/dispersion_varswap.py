"""
Dispersion strategy using variance swap replication on each leg.

Same dispersion logic (index vs components, direction flipping, 1/N scaling)
but each leg is an OTM option strip weighted by 2*dK/K^2 (Demeterfi et al.)
instead of a single straddle/strangle.

Long dispersion (direction=1):
  - Buy component variance, sell index variance
  - Profits when realized correlation < implied correlation

Short dispersion (direction=-1):
  - Sell component variance, buy index variance
"""

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class DispersionVarSwapStrategy(Strategy):
    """Dispersion trading with variance swap replication on each leg."""

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
        disp_dir = cfg.dispersion_direction
        if is_index:
            trade_direction = -disp_dir
        else:
            trade_direction = disp_dir
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
        exp_dte = (
            chain[["expiration", "dte"]]
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

        # Filter chain to this expiry
        sub = chain[chain["expiration"] == expiry].copy()
        if sub.empty:
            return

        if "delta" not in sub.columns:
            return

        # --- Build OTM strip (Demeterfi et al.) ---
        delta_cutoff = cfg.varswap_delta_cutoff
        delta_max = cfg.varswap_delta_max

        # OTM puts: K < spot, cutoff <= |delta| <= delta_max
        puts = sub[
            (sub["type"] == "P")
            & (sub["strike"] < spot)
            & (sub["delta"].abs() >= delta_cutoff)
            & (sub["delta"].abs() <= delta_max)
        ].copy()

        # OTM calls: K >= spot, cutoff <= |delta| <= delta_max
        calls = sub[
            (sub["type"] == "C")
            & (sub["strike"] >= spot)
            & (sub["delta"].abs() >= delta_cutoff)
            & (sub["delta"].abs() <= delta_max)
        ].copy()

        strip = pd.concat([puts, calls], ignore_index=True)
        strip = strip.sort_values("strike").reset_index(drop=True)

        if len(strip) < 2:
            return

        strip = strip[strip["vega"].abs() > 1e-12].reset_index(drop=True)
        if len(strip) < 2:
            return

        # --- Compute 1/K^2 weights ---
        strikes = strip["strike"].values
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

        # --- Size to vega target ---
        vegas = strip["vega"].values.astype(float)
        weighted_vega = np.sum(weights * np.abs(vegas))
        if weighted_vega < 1e-12:
            return

        # Build legs
        legs: List[Dict[str, Any]] = []
        for i in range(n):
            row = strip.iloc[i]
            qty = trade_direction * abs(vega_target) * weights[i] / weighted_vega
            if abs(qty) < 1e-10:
                continue
            legs.append({
                "contract_id": row["contractID"],
                "expiry": expiry_date,
                "strike": float(row["strike"]),
                "type": row["type"],
                "qty": float(qty),
            })

        if not legs:
            return

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
                "mode": "dispersion_varswap",
                "role": "index" if is_index else "component",
                "direction": float(trade_direction),
                "n_legs": len(legs),
                "target_dte": target_dte,
                "expiry_dte": int(exp_dte.loc[expiry]),
            },
        )
