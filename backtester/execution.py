from typing import Dict

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .models import TickerState


class ExecutionModel:
    """Handles trade pricing and cash booking for option and stock trades."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def get_option_trade_price(
        self, row_opt: pd.Series, trade_qty: float, mid: float
    ) -> float:
        mode = str(self.config.execution_mode).lower()

        if mode == "mid":
            return mid

        if mode in ("bid_ask", "bidask"):
            bid = float(row_opt.get("bid", np.nan))
            ask = float(row_opt.get("ask", np.nan))
            if trade_qty > 0:
                return ask if np.isfinite(ask) and ask > 0 else mid
            return bid if np.isfinite(bid) and bid > 0 else mid

        # Default: "mid_spread" - apply bps spread to mid
        option_spread_bps = float(self.config.cost_model.get("option_spread_bps", 0.0))
        spread = option_spread_bps / 1e4
        return mid * (1 + spread) if trade_qty > 0 else mid * (1 - spread)

    def book_option_trade(
        self,
        st: TickerState,
        trade_qty: float,
        trade_price: float,
        mid_price: float,
    ) -> Dict[str, float]:
        commission_per_contract = float(
            self.config.cost_model.get("commission_per_contract", 0.0)
        )

        cash_change = -trade_qty * trade_price
        st.cash += cash_change

        tc_spread = abs(trade_price - mid_price) * abs(trade_qty)
        tc_comm = commission_per_contract * abs(trade_qty)
        st.cash -= tc_comm

        return {
            "cash_change": cash_change,
            "tc_spread": tc_spread,
            "tc_comm": tc_comm,
        }

    def apply_delta_hedge(
        self,
        st: TickerState,
        port_delta_today: float,
        spot: float,
    ) -> Dict[str, float]:
        if not self.config.delta_hedge:
            st.stock_pos_intraday = st.stock_pos_close
            return {"pnl_delta_hedge": 0.0, "pnl_tc_stock": 0.0}

        # Guard against NaN delta (safety net if NaN leaks through)
        if not np.isfinite(port_delta_today):
            port_delta_today = 0.0
        tgt_stock_pos = -port_delta_today
        prev_stock = st.stock_pos_close if np.isfinite(st.stock_pos_close) else 0.0
        trade_shares = tgt_stock_pos - prev_stock
        pnl_delta_hedge = 0.0
        pnl_tc_stock = 0.0

        stock_spread_bps = float(self.config.cost_model.get("stock_spread_bps", 0.0))

        if abs(trade_shares) > 1e-10:
            spread = stock_spread_bps / 1e4
            trade_price = spot * (1 + spread) if trade_shares > 0 else spot * (1 - spread)
            cash_change = -trade_shares * trade_price
            st.cash += cash_change

            tc_stock = abs(trade_price - spot) * abs(trade_shares)
            pnl_tc_stock += tc_stock

        if st.last_spot is not None:
            dS = spot - st.last_spot
            pnl_delta_hedge = st.stock_pos_close * dS

        st.stock_pos_intraday = tgt_stock_pos
        st.stock_pos_close = tgt_stock_pos

        return {"pnl_delta_hedge": pnl_delta_hedge, "pnl_tc_stock": pnl_tc_stock}
