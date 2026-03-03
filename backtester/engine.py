from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from .config import BacktestConfig
from .data_store import DataStore
from .metrics import BacktestMetrics, compute_metrics
from .portfolio import PortfolioEngine
from .reporting import export_results
from .strategies import EarningsStrategy, RollingStrategy, Strategy
from .strategies.variance_swap import VarianceSwapStrategy
from .strategies.calendar_spread import CalendarSpreadStrategy
from .strategies.dispersion import DispersionStrategy
from .strategies.dispersion_varswap import DispersionVarSwapStrategy

# Strategy registry: add new strategies here
STRATEGY_REGISTRY = {
    "earnings": EarningsStrategy,
    "rolling": RollingStrategy,
    "varswap": VarianceSwapStrategy,
    "calendar": CalendarSpreadStrategy,
    "dispersion": DispersionStrategy,
    "dispersion_varswap": DispersionVarSwapStrategy,
}


class BacktestEngine:
    """Main orchestrator: builds components and runs the backtest loop."""

    def __init__(self, config: BacktestConfig, market: Optional[DataStore] = None):
        self.config = config
        self._owns_market = market is None
        self.market = market if market is not None else DataStore(config)
        self.strategy = self._build_strategy(config)
        self.portfolio = PortfolioEngine(
            config, config.symbols, self.market, self.strategy
        )
        self.strategy.initialize(self.market)

    @staticmethod
    def _build_strategy(config: BacktestConfig) -> Strategy:
        cls = STRATEGY_REGISTRY.get(config.strategy_mode)
        if cls is None:
            raise ValueError(
                f"Unknown strategy_mode: {config.strategy_mode}. "
                f"Available: {list(STRATEGY_REGISTRY.keys())}"
            )
        return cls(config)

    def run(self) -> BacktestMetrics:
        """Execute the backtest and return performance metrics."""
        all_dates = self._build_global_calendar()
        print(f"[INFO] Backtest calendar: {len(all_dates)} days.")

        for sym in self.config.symbols:
            self.market.load_symbol_options(sym)

            for date in tqdm(all_dates, desc=f"Processing {sym}", unit="day"):
                if date not in self.market.get_calendar(sym):
                    continue
                self.portfolio.process_symbol_date(sym, date)

            # Free memory after processing each symbol (only if we own the DataStore)
            if self._owns_market and sym in self.market.options:
                del self.market.options[sym]

        # Compute metrics
        daily_df = pd.DataFrame(self.portfolio.daily_pnl_rows)

        # Build event PnL for metrics
        event_df = self._build_event_pnl(daily_df)

        metrics = compute_metrics(daily_df, event_df)

        # Export
        export_results(
            daily_df=daily_df,
            trade_rows=self.portfolio.trade_rows,
            config=self.config,
            market=self.market,
            strategy=self.strategy,
            metrics=metrics,
            all_ptfs=self.portfolio.all_ptfs,
        )

        return metrics

    def _build_global_calendar(self) -> pd.DatetimeIndex:
        calendars = [self.market.get_calendar(sym) for sym in self.config.symbols]
        if not calendars:
            return pd.DatetimeIndex([])
        all_dates = sorted(set().union(*[set(idx) for idx in calendars]))
        return pd.DatetimeIndex(all_dates)

    def _build_event_pnl(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Build event-level PnL DataFrame from daily data and entry_exit_map."""
        if daily_df.empty:
            return pd.DataFrame()

        entry_exit_map = getattr(self.strategy, "entry_exit_map", {})
        if not entry_exit_map:
            return pd.DataFrame()

        event_rows = []
        for sym, mapping in entry_exit_map.items():
            for entry_date, meta in mapping.items():
                exit_date = meta.get("exit_date")
                if exit_date is None:
                    continue
                mask = (
                    (daily_df["Symbol"] == sym)
                    & (daily_df["Date"] >= entry_date)
                    & (daily_df["Date"] <= exit_date)
                )
                window = daily_df[mask]
                if window.empty:
                    continue
                pnl = window["DailyPnL"].sum()
                # Skip events where no trade was placed (signal filter, missing data, etc.)
                if abs(pnl) < 1e-10:
                    continue
                event_rows.append({
                    "Symbol": sym,
                    "EntryDate": entry_date,
                    "EventDay": meta["event_day"],
                    "ExitDate": exit_date,
                    "Timing": meta.get("timing", "UNKNOWN"),
                    "EventWindowPnL": pnl,
                })

        return pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
