"""
Discrete variance swap replication using a strip of listed OTM options.

Theory (Demeterfi et al. 1999):
    Var_swap ≈ (2/T) * sum_i [ (dK_i / K_i^2) * Price_i ]

    where the sum runs over OTM puts (K < F) and OTM calls (K >= F),
    weighted by 1/K^2 * dK.  This creates a portfolio with approximately
    constant dollar-gamma, replicating the variance swap payoff.

Default: enter short every Friday, expire next Friday (weekly).
"""

from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class VarianceSwapStrategy(Strategy):
    """Variance swap replication via OTM option strip."""

    def __init__(self, config: BacktestConfig):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Sizing: same as default (scale with perf), but can be overridden
    # ------------------------------------------------------------------
    def compute_vega_target(self, state: TickerState) -> float:
        cfg = self.config
        perf_scale = state.perf / cfg.initial_perf_per_ticker if cfg.reinvest else 1.0
        return cfg.base_vega_target * perf_scale

    # ------------------------------------------------------------------
    # Entry frequency
    # ------------------------------------------------------------------
    def _is_entry_day(self, date: pd.Timestamp) -> bool:
        freq = self.config.varswap_entry_frequency
        if freq == "weekly":
            return date.weekday() == self.config.varswap_entry_weekday
        elif freq == "monthly_opex":
            return self._is_third_friday(date)
        return True

    @staticmethod
    def _is_third_friday(date: pd.Timestamp) -> bool:
        if date.weekday() != 4:
            return False
        return 15 <= date.day <= 21

    # ------------------------------------------------------------------
    # Core: build the option strip on entry day
    # ------------------------------------------------------------------
    def on_day(self, date, symbol, state, market, portfolio, vega_target) -> None:
        if not self._is_entry_day(date):
            return

        if vega_target is None or abs(vega_target) < 1e-12:
            return

        chain = market.get_chain(symbol, date)
        if chain is None or chain.empty:
            return

        # Normalize column names
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

        # --- Select expiry: nearest >= target_dte ---
        target_dte = self.config.varswap_target_dte
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

        # Need delta column for cutoff
        if "delta" not in sub.columns:
            return

        # --- Build OTM strip ---
        delta_cutoff = self.config.varswap_delta_cutoff
        delta_max = self.config.varswap_delta_max

        # OTM puts: K < spot, delta < 0, cutoff <= |delta| <= delta_max
        puts = sub[
            (sub["type"] == "P")
            & (sub["strike"] < spot)
            & (sub["delta"].abs() >= delta_cutoff)
            & (sub["delta"].abs() <= delta_max)
        ].copy()

        # OTM calls: K >= spot, delta > 0, cutoff <= |delta| <= delta_max
        calls = sub[
            (sub["type"] == "C")
            & (sub["strike"] >= spot)
            & (sub["delta"].abs() >= delta_cutoff)
            & (sub["delta"].abs() <= delta_max)
        ].copy()

        # Combine and sort by strike
        strip = pd.concat([puts, calls], ignore_index=True)
        strip = strip.sort_values("strike").reset_index(drop=True)

        if len(strip) < 2:
            return

        # Filter for valid vega
        strip = strip[strip["vega"].abs() > 1e-12].reset_index(drop=True)
        if len(strip) < 2:
            return

        # --- Compute 1/K^2 weights ---
        strikes = strip["strike"].values
        n = len(strikes)

        # dK: strike spacing (central difference for interior, one-sided for edges)
        dk = np.zeros(n)
        for i in range(n):
            if i == 0:
                dk[i] = strikes[1] - strikes[0]
            elif i == n - 1:
                dk[i] = strikes[-1] - strikes[-2]
            else:
                dk[i] = (strikes[i + 1] - strikes[i - 1]) / 2.0

        # Weight: 2 * dK / K^2  (the factor 2 comes from the variance formula)
        weights = 2.0 * dk / (strikes ** 2)

        # --- Size to vega target ---
        vegas = strip["vega"].values.astype(float)
        weighted_vega = np.sum(weights * np.abs(vegas))
        if weighted_vega < 1e-12:
            return

        direction = self.config.varswap_direction  # -1 for short variance

        # Build legs
        legs: List[Dict[str, Any]] = []
        for i in range(n):
            row = strip.iloc[i]
            qty = direction * abs(vega_target) * weights[i] / weighted_vega
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

        # Exit at expiry
        portfolio.register_new_ptf(
            symbol=symbol,
            entry_date=date.normalize(),
            exit_date=expiry_date,
            legs=legs,
            meta={
                "mode": "varswap",
                "n_legs": len(legs),
                "direction": direction,
                "target_dte": target_dte,
                "expiry_dte": int(exp_dte.loc[expiry]),
            },
        )
