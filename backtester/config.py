from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import pandas as pd


@dataclass
class BacktestConfig:
    # Universe & dates
    symbols: List[str] = field(default_factory=list)
    start_date: pd.Timestamp = pd.Timestamp("2020-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-11-16")

    # Paths
    corp_dir: str = "alpha_corp_actions"
    options_dir: str = "alpha_options_raw"
    earnings_csv: str = "earnings.csv"
    output_path: str = "outputs/backtest.xlsx"

    # Portfolio
    initial_perf_per_ticker: float = 100.0
    reinvest: bool = True
    base_vega_target: float = 1.0 / 20

    # Strategy routing: "earnings", "rolling", or "varswap"
    strategy_mode: str = "earnings"

    # --- Earnings-specific ---
    earnings_structure: str = "straddle"  # "straddle" or "strangle"
    strangle_mny_offset: float = 0.035
    entry_lag: Dict[str, int] = field(
        default_factory=lambda: {"BMO": -1, "AMC": 0, "DURING": 0, "UNKNOWN": 0}
    )
    exit_lag: Dict[str, int] = field(
        default_factory=lambda: {"BMO": 0, "AMC": 1, "DURING": 1, "UNKNOWN": 1}
    )

    # --- Rolling-specific ---
    rolling_leg_type: str = "P"
    rolling_select_by: str = "delta"  # "delta" or "moneyness"
    rolling_target_delta: float = -0.20
    rolling_target_mny: float = 1.0
    rolling_maturity_index: int = 3
    rolling_expiry_selection: Dict[str, Any] = field(
        default_factory=lambda: {"mode": "min_dte", "min_dte": 20}
    )
    rolling_min_dte: int = 1
    rolling_max_dte: int = 365
    rolling_holding_days: int = 20
    rolling_vega_per_lot: Optional[float] = None
    rolling_reinvest: bool = False
    rolling_direction: float = 1.0
    rolling_entry_frequency: str = "daily"  # "daily", "weekly", "monthly_opex"
    rolling_entry_weekday: int = 4  # 0=Mon..4=Fri. Used when frequency="weekly"

    # --- Variance Swap replication ---
    varswap_target_dte: int = 7
    varswap_delta_cutoff: float = 0.05
    varswap_delta_max: float = 1.0        # max |delta| for strip (1.0 = no cap)
    varswap_entry_frequency: str = "weekly"  # "weekly" or "monthly_opex"
    varswap_entry_weekday: int = 4  # 0=Mon..4=Fri
    varswap_direction: float = -1.0  # -1 = short variance, +1 = long variance

    # --- Calendar Spread ---
    calendar_front_dte: int = 7           # target DTE for front (short) leg
    calendar_back_dte: int = 30           # target DTE for back (long) leg
    calendar_leg_type: str = "C"          # "C", "P", or "straddle"
    calendar_select_by: str = "atm"       # "atm", "delta", or "moneyness"
    calendar_target_delta: float = 0.50   # used when select_by="delta"
    calendar_target_mny: float = 1.0      # used when select_by="moneyness"
    calendar_direction: float = 1.0       # 1.0 = long calendar, -1.0 = short calendar
    calendar_entry_frequency: str = "weekly"  # "daily", "weekly", "monthly_opex"
    calendar_entry_weekday: int = 4       # 0=Mon..4=Fri

    # --- Dispersion ---
    dispersion_index_symbol: str = "SPY"
    dispersion_component_symbols: List[str] = field(default_factory=list)
    dispersion_direction: float = 1.0     # 1.0 = long dispersion, -1.0 = short
    dispersion_structure: str = "straddle"  # "straddle" or "strangle"
    dispersion_target_dte: int = 30
    dispersion_holding_days: int = 20
    dispersion_entry_frequency: str = "weekly"  # "daily", "weekly", "monthly_opex"
    dispersion_entry_weekday: int = 4     # 0=Mon..4=Fri
    dispersion_shares_outstanding: Dict[str, float] = field(default_factory=dict)  # shares (B) for mkt-cap weighting

    # --- Signals ---
    use_signal: bool = False
    signal_mode: str = "long"  # "short", "long", "ls"
    signal_csv_path: str = "outputs/signals_all.csv"
    signal_min_years: int = 2
    signal_n_bins: int = 10
    signal_max_vega_mult: float = 2.0

    # --- Simple threshold filter (overrides walk-forward bins when set) ---
    signal_filter_col: str = ""       # e.g. "TermSlope_pre_z" — skip events below threshold
    signal_filter_min: float = 0.0    # minimum value in signal_filter_col to trade

    # --- PnL Explain (independent-bump BS repricing attribution) ---
    pnl_explain: bool = False         # enable BS repricing PnL attribution
    pnl_explain_n_steps: int = 10     # N-step subdivision for spot/gamma decomposition

    # --- Execution ---
    execution_mode: str = "mid_spread"  # "mid_spread", "mid", "bid_ask"
    cost_model: Dict[str, float] = field(
        default_factory=lambda: {
            "option_spread_bps": 50,
            "stock_spread_bps": 1,
            "commission_per_contract": 0.0,
        }
    )
    delta_hedge: bool = True

    # --- Data quality filters ---
    min_moneyness: float = 0.5
    max_moneyness: float = 1.5
    min_dte_for_entry: int = 5
    max_dte_for_entry: int = 30
    min_open_interest: int = 0
    min_volume: int = 0
    max_bid_ask_spread_pct: float = 0.5

    # --- Options loading optimization ---
    optimized_options_loading: bool = True   # False = legacy full-chain load
    options_cache_dir: str = "cache_options" # where pre-processed parquets live

    # --- Exit fallback ---
    exit_fallback_mode: str = "intrinsic"  # "intrinsic" or "next"
    exit_fallback_max_wait_bdays: int = 3
    trade_log_mode: str = "all"  # "all", "entries", "light" (first 2 rows), or "summary" (none)

    def to_flat_dict(self) -> List[Dict[str, Any]]:
        """Flatten config to list of {Key, Value} rows for Excel export."""
        return _flatten_config_dict(self.__dict__)


def _flatten_config_dict(d: dict, prefix: str = "") -> List[Dict[str, Any]]:
    rows = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            rows.extend(_flatten_config_dict(v, prefix=key))
        else:
            rows.append({"Key": key, "Value": v})
    return rows
