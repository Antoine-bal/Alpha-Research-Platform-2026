from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState
from .base import Strategy


class RollingStrategy(Strategy):
    """
    Daily rolling single-leg option strategy.

    Every business day:
      - Select an expiry based on rolling_expiry_selection config
      - Pick one leg (P or C) by delta or moneyness
      - Size to vega_target * rolling_direction
      - Hold for rolling_holding_days, then close
    """

    def __init__(self, config: BacktestConfig):
        super().__init__(config)

    def compute_vega_target(self, state) -> float:
        cfg = self.config
        holding_days = max(1, cfg.rolling_holding_days)
        rolling_vega_per_ptf = float(
            cfg.rolling_vega_per_lot or (cfg.base_vega_target / holding_days)
        )
        perf_scale = state.perf / cfg.initial_perf_per_ticker if cfg.rolling_reinvest else 1.0
        return rolling_vega_per_ptf * perf_scale

    def _is_entry_day(self, date: pd.Timestamp) -> bool:
        """Check if today is a valid entry day per rolling_entry_frequency."""
        freq = self.config.rolling_entry_frequency
        if freq == "daily":
            return True
        elif freq == "weekly":
            return date.weekday() == self.config.rolling_entry_weekday
        elif freq == "monthly_opex":
            return self._is_third_friday(date)
        return True

    @staticmethod
    def _is_third_friday(date: pd.Timestamp) -> bool:
        """Check if date is the 3rd Friday of its month."""
        if date.weekday() != 4:
            return False
        return 15 <= date.day <= 21

    def on_day(self, date, symbol, state, market, portfolio, vega_target) -> None:
        cfg = self.config

        # Entry frequency gate
        if not self._is_entry_day(date):
            return

        if vega_target is None or abs(vega_target) < 1e-12:
            return

        chain = market.get_chain(symbol, date)
        if chain is None or chain.empty:
            return

        if "expiration" not in chain.columns:
            if "expiry" in chain.columns:
                chain = chain.rename(columns={"expiry": "expiration"})
            else:
                return

        if "dte" not in chain.columns:
            chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.normalize()
            chain["dte"] = (chain["expiration"] - date.normalize()).dt.days

        min_dte = cfg.rolling_min_dte
        max_dte = cfg.rolling_max_dte
        chain = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]
        if chain.empty:
            return

        # Filter by option type
        leg_type = cfg.rolling_leg_type.upper()[0]
        chain = chain[chain["type"] == leg_type]
        if chain.empty:
            return

        # Pick maturity
        expiries = sorted(chain["expiration"].unique())
        if not expiries:
            return

        expiry_sel = cfg.rolling_expiry_selection or {}
        sel_mode = str(expiry_sel.get("mode") or "").lower()
        target_dte = expiry_sel.get("target_dte")
        band_min = expiry_sel.get("min_dte")
        band_max = expiry_sel.get("max_dte")

        expiry = None
        if sel_mode in ("closest_dte", "min_dte", "band") and target_dte is not None:
            exp_dte = (
                chain[["expiration", "dte"]]
                .dropna()
                .groupby("expiration")["dte"]
                .min()
            )
            if sel_mode == "closest_dte":
                expiry = exp_dte.sub(float(target_dte)).abs().idxmin()
            elif sel_mode == "min_dte":
                eligible = exp_dte[exp_dte >= float(target_dte)]
                expiry = (
                    eligible.idxmin()
                    if not eligible.empty
                    else exp_dte.sub(float(target_dte)).abs().idxmin()
                )
            elif sel_mode == "band":
                use_min = float(band_min) if band_min is not None else float(cfg.rolling_min_dte)
                use_max = float(band_max) if band_max is not None else float(cfg.rolling_max_dte)
                eligible = exp_dte[(exp_dte >= use_min) & (exp_dte <= use_max)]
                if not eligible.empty:
                    expiry = eligible.sub(float(target_dte)).abs().idxmin()
                else:
                    expiry = exp_dte.sub(float(target_dte)).abs().idxmin()

        if expiry is None:
            idx = min(cfg.rolling_maturity_index, len(expiries) - 1)
            expiry = expiries[idx]

        sub = chain[chain["expiration"] == expiry]
        if sub.empty:
            return

        # Select strike
        select_by = cfg.rolling_select_by
        spot_today = market.get_spot(symbol, date)
        if spot_today is None or not np.isfinite(spot_today):
            return

        if select_by == "moneyness":
            if "moneyness" not in sub.columns:
                if "strike_eff" in sub.columns:
                    sub["moneyness"] = sub["strike_eff"] / spot_today
                else:
                    sub["moneyness"] = sub["strike"] / spot_today

        if select_by == "delta":
            if "delta" not in sub.columns:
                return
            target_delta = cfg.rolling_target_delta
            delta_diff = (sub["delta"] - target_delta).abs().to_numpy()
            chosen = sub.iloc[int(delta_diff.argmin())]
        elif select_by == "moneyness":
            target_mny = cfg.rolling_target_mny
            mny_diff = (sub["moneyness"] - target_mny).abs().to_numpy()
            chosen = sub.iloc[int(mny_diff.argmin())]
        else:
            if "moneyness" not in sub.columns:
                if "strike_eff" in sub.columns:
                    sub["moneyness"] = sub["strike_eff"] / spot_today
                else:
                    sub["moneyness"] = sub["strike"] / spot_today
            mny_diff = (sub["moneyness"] - 1.0).abs().to_numpy()
            chosen = sub.iloc[int(mny_diff.argmin())]

        # Vega sizing
        if "vega" not in chosen.index:
            return
        vega_leg = float(chosen["vega"])
        if not np.isfinite(vega_leg) or abs(vega_leg) < 1e-12:
            return

        direction = cfg.rolling_direction
        total_vega = direction * abs(vega_target)
        qty = total_vega / vega_leg

        if abs(qty) < 1e-8:
            return

        # Exit date
        holding_days = max(1, cfg.rolling_holding_days)
        bdays = pd.bdate_range(start=date.normalize(), periods=holding_days)
        target_close_date = bdays[-1]

        expiry_date = pd.to_datetime(chosen["expiration"]).normalize()
        exit_date = min(expiry_date, target_close_date)

        leg_dict = {
            "contract_id": chosen["contractID"],
            "expiry": expiry_date,
            "strike": float(chosen["strike"]),
            "type": chosen["type"],
            "qty": float(qty),
        }

        portfolio.register_new_ptf(
            symbol=symbol,
            entry_date=date.normalize(),
            exit_date=exit_date,
            legs=[leg_dict],
            meta={
                "mode": "rolling",
                "leg_type": leg_type,
                "select_by": select_by,
                "target_delta": cfg.rolling_target_delta,
                "target_mny": cfg.rolling_target_mny,
                "holding_days": holding_days,
            },
        )
