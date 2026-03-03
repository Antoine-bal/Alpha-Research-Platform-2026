from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from ..config import BacktestConfig
from ..models import TickerState

if TYPE_CHECKING:
    from ..data_store import DataStore
    from ..portfolio import PortfolioEngine


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def compute_vega_target(self, state: TickerState) -> float:
        """Compute the vega target for today. Override in subclasses.

        Default: scale base_vega_target linearly with performance.
        """
        cfg = self.config
        perf_scale = state.perf / cfg.initial_perf_per_ticker if cfg.reinvest else 1.0
        return cfg.base_vega_target * perf_scale

    def initialize(self, market: "DataStore") -> None:
        """Called once before the backtest loop starts.

        Use this to build entry/exit maps, load signals, pre-compute
        anything that depends on market data.
        """
        pass

    def adjust_hedge_delta(self, live_ptfs, chain_idx, spot, bs_delta_total):
        """Hook for strategies to adjust the aggregate hedge delta.

        Called after the MTM loop computes the BS delta across all live legs.
        Return an additive correction (e.g. skew delta). Default: 0 (no adjustment).
        """
        return 0.0

    @abstractmethod
    def on_day(
        self,
        date: pd.Timestamp,
        symbol: str,
        state: TickerState,
        market: "DataStore",
        portfolio: "PortfolioEngine",
        vega_target: float,
    ) -> None:
        """Called once per (date, symbol) during the backtest loop.

        The strategy should call portfolio.register_new_ptf() to open
        new positions when appropriate. It should NOT directly modify
        state or market data.
        """
        ...
