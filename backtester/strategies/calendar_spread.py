"""
Calendar spread strategy: sell short-dated option, buy long-dated option
at the same (or nearby) strike.

Long calendar (direction=1):  buy back month, sell front month
  - Net positive vega
  - Benefits from vol increase and front-month time decay

Short calendar (direction=-1): sell back month, buy front month
  - Net negative vega

Exit: at front expiry (close the back leg).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class CalendarSpreadStrategy(Strategy):
    """Calendar spread (time spread) strategy."""

    def __init__(self, config: BacktestConfig):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Entry frequency
    # ------------------------------------------------------------------
    def _is_entry_day(self, date: pd.Timestamp) -> bool:
        freq = self.config.calendar_entry_frequency
        if freq == "daily":
            return True
        elif freq == "weekly":
            return date.weekday() == self.config.calendar_entry_weekday
        elif freq == "monthly_opex":
            return self._is_third_friday(date)
        return True

    @staticmethod
    def _is_third_friday(date: pd.Timestamp) -> bool:
        if date.weekday() != 4:
            return False
        return 15 <= date.day <= 21

    # ------------------------------------------------------------------
    # Strike selection
    # ------------------------------------------------------------------
    def _select_strike(
        self, sub: pd.DataFrame, spot: float
    ) -> Optional[float]:
        cfg = self.config
        select_by = cfg.calendar_select_by

        if select_by == "delta":
            if "delta" not in sub.columns:
                return None
            target = cfg.calendar_target_delta
            delta_diff = (sub["delta"] - target).abs().to_numpy()
            return float(sub.iloc[int(delta_diff.argmin())]["strike"])

        # moneyness or atm
        if "moneyness" not in sub.columns or sub["moneyness"].isna().all():
            if "strike_eff" in sub.columns:
                mny = sub["strike_eff"] / spot
            else:
                mny = sub["strike"] / spot
        else:
            mny = sub["moneyness"]

        target = cfg.calendar_target_mny if select_by == "moneyness" else 1.0
        mny_diff = (mny - target).abs().to_numpy()
        return float(sub.iloc[int(mny_diff.argmin())]["strike"])

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def on_day(self, date, symbol, state, market, portfolio, vega_target) -> None:
        cfg = self.config

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

        # --- Find front and back expiries ---
        front_target_dte = cfg.calendar_front_dte
        back_target_dte = cfg.calendar_back_dte

        exp_dte = (
            chain[["expiration", "dte"]]
            .drop_duplicates()
            .groupby("expiration")["dte"]
            .min()
        )

        # Front: nearest expiry >= front_target_dte
        front_eligible = exp_dte[exp_dte >= front_target_dte]
        if front_eligible.empty:
            return
        front_expiry = front_eligible.idxmin()
        front_expiry_date = pd.to_datetime(front_expiry).normalize()

        # Back: nearest expiry >= back_target_dte AND strictly after front
        back_eligible = exp_dte[
            (exp_dte >= back_target_dte) & (exp_dte.index > front_expiry)
        ]
        if back_eligible.empty:
            return
        back_expiry = back_eligible.idxmin()
        back_expiry_date = pd.to_datetime(back_expiry).normalize()

        if front_expiry_date >= back_expiry_date:
            return

        # --- Select option type(s) ---
        leg_type = cfg.calendar_leg_type.upper()
        if leg_type == "STRADDLE":
            types_to_trade = ["C", "P"]
        else:
            types_to_trade = [leg_type[0]]

        direction = cfg.calendar_direction  # 1.0 = long calendar

        all_leg_info: List[Dict[str, Any]] = []
        total_net_vega = 0.0

        for opt_type in types_to_trade:
            back_chain = chain[
                (chain["expiration"] == back_expiry) & (chain["type"] == opt_type)
            ]
            front_chain = chain[
                (chain["expiration"] == front_expiry) & (chain["type"] == opt_type)
            ]

            if back_chain.empty or front_chain.empty:
                continue

            # Select strike from back-month chain
            strike = self._select_strike(back_chain, spot)
            if strike is None:
                continue

            # Find closest strike in both chains
            back_idx = int((back_chain["strike"] - strike).abs().to_numpy().argmin())
            front_idx = int((front_chain["strike"] - strike).abs().to_numpy().argmin())
            back_row = back_chain.iloc[back_idx]
            front_row = front_chain.iloc[front_idx]

            back_vega = float(back_row.get("vega", 0.0))
            front_vega = float(front_row.get("vega", 0.0))
            if abs(back_vega) < 1e-12 or abs(front_vega) < 1e-12:
                continue

            # Net vega per unit = back_vega - front_vega (positive for long calendar)
            net_vega = back_vega - front_vega
            total_net_vega += abs(net_vega)

            all_leg_info.append({
                "back_row": back_row,
                "front_row": front_row,
                "opt_type": opt_type,
                "net_vega": net_vega,
            })

        if not all_leg_info or total_net_vega < 1e-12:
            return

        # Size so that |net portfolio vega| = |vega_target|
        qty = direction * abs(vega_target) / total_net_vega
        if abs(qty) < 1e-10:
            return

        # Build legs
        legs: List[Dict[str, Any]] = []
        for info in all_leg_info:
            # Back leg: buy (qty > 0 for long calendar)
            legs.append({
                "contract_id": info["back_row"]["contractID"],
                "expiry": back_expiry_date,
                "strike": float(info["back_row"]["strike"]),
                "type": info["opt_type"],
                "qty": float(qty),
            })
            # Front leg: sell (qty < 0 for long calendar)
            legs.append({
                "contract_id": info["front_row"]["contractID"],
                "expiry": front_expiry_date,
                "strike": float(info["front_row"]["strike"]),
                "type": info["opt_type"],
                "qty": float(-qty),
            })

        if not legs:
            return

        # Exit at front expiry (close the back leg)
        exit_date = front_expiry_date

        portfolio.register_new_ptf(
            symbol=symbol,
            entry_date=date.normalize(),
            exit_date=exit_date,
            legs=legs,
            meta={
                "mode": "calendar",
                "direction": direction,
                "front_dte": int(exp_dte.loc[front_expiry]),
                "back_dte": int(exp_dte.loc[back_expiry]),
                "leg_type": leg_type,
                "n_legs": len(legs),
            },
        )
