from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd


@dataclass
class OptionLeg:
    contract_id: str
    expiry: pd.Timestamp
    strike: float
    opt_type: str
    qty: float
    prev_price: float = 0.0
    prev_iv: float = 0.0
    prev_delta: float = 0.0
    prev_gamma: float = 0.0
    prev_vega: float = 0.0
    prev_theta: float = 0.0


@dataclass
class RollingPtf:
    """
    One 'portfolio slice' opened on entry_date and closed on exit_date.
    May contain several contracts (legs), all opened at entry_date and
    all closed at exit_date (unless they hit expiry earlier).
    """
    ptf_id: int
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    legs: List[OptionLeg]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TickerState:
    perf: float
    cash: float

    rolling_ptfs: Dict[int, RollingPtf] = field(default_factory=dict)
    next_ptf_id: int = 1

    stock_pos_close: float = 0.0
    stock_pos_intraday: float = 0.0

    last_spot: Optional[float] = None
    last_date: Optional[pd.Timestamp] = None
    mtm_options: float = 0.0
    mtm_stock: float = 0.0

    cum_pnl: float = 0.0
    cum_pnl_gamma: float = 0.0
    cum_pnl_vega: float = 0.0
    cum_pnl_theta: float = 0.0
    cum_pnl_rho: float = 0.0
    cum_pnl_delta_hedge: float = 0.0
    cum_pnl_tc: float = 0.0
    cum_pnl_residual: float = 0.0
