from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .models import OptionLeg, RollingPtf, TickerState
from .execution import ExecutionModel
from bs_pricing import bs_price, bs_greeks


class PortfolioEngine:
    """Manages positions, mark-to-market, splits, and delta hedging."""

    def __init__(
        self,
        config: BacktestConfig,
        symbols: List[str],
        market,  # DataStore
        strategy,  # Strategy
    ):
        self.config = config
        self.symbols = symbols
        self.market = market
        self.strategy = strategy
        self.execution = ExecutionModel(config)

        self.daily_pnl_rows: List[Dict[str, Any]] = []
        self.trade_rows: List[Dict[str, Any]] = []

        # Global registry of all portfolios
        self.all_ptfs: Dict[int, Dict[str, Any]] = {}
        self._next_ptf_id: int = 1

        self.state: Dict[str, TickerState] = {
            sym: TickerState(
                perf=config.initial_perf_per_ticker,
                cash=config.initial_perf_per_ticker,
            )
            for sym in symbols
        }

    def register_new_ptf(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        legs: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        ptf_id = self._next_ptf_id
        self._next_ptf_id += 1

        leg_objs = [
            OptionLeg(
                contract_id=str(leg["contract_id"]),
                expiry=pd.to_datetime(leg["expiry"]).normalize(),
                strike=float(leg["strike"]),
                opt_type=str(leg["type"]).upper()[0],
                qty=float(leg["qty"]),
            )
            for leg in legs
        ]

        meta = meta or {}

        meta_row = {
            "ptf_id": ptf_id,
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "contract_ids": [leg.contract_id for leg in leg_objs],
            "expiries": [leg.expiry for leg in leg_objs],
            "strikes": [leg.strike for leg in leg_objs],
            "opt_types": [leg.opt_type for leg in leg_objs],
            "qtys": [leg.qty for leg in leg_objs],
            "meta": meta,
        }
        self.all_ptfs[ptf_id] = meta_row

        st = self.state[symbol]
        ptf = RollingPtf(
            ptf_id=ptf_id,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            legs=leg_objs,
            meta=meta,
        )
        st.rolling_ptfs[ptf_id] = ptf
        return ptf_id

    def _apply_split_to_ptfs(self, symbol: str, date: pd.Timestamp) -> None:
        split_factor = self.market.get_split_factor(symbol, date)

        if not np.isfinite(split_factor) or abs(split_factor - 1.0) < 1e-12:
            return

        st = self.state[symbol]

        for ptf_id, ptf in st.rolling_ptfs.items():
            if not (ptf.entry_date <= date <= ptf.exit_date):
                continue
            if not ptf.legs:
                continue

            for leg in ptf.legs:
                leg.strike = float(leg.strike) / split_factor
                leg.qty = float(leg.qty) * split_factor
                leg.prev_delta = 0.0
                leg.prev_gamma = 0.0
                leg.prev_vega = 0.0
                leg.prev_theta = 0.0

            meta = self.all_ptfs.get(ptf_id)
            if meta is not None:
                meta["strikes"] = [leg.strike for leg in ptf.legs]
                meta["qtys"] = [leg.qty for leg in ptf.legs]

        st.stock_pos_close *= split_factor
        st.stock_pos_intraday *= split_factor

        if st.last_spot is not None and np.isfinite(st.last_spot):
            st.last_spot = st.last_spot / split_factor

    def _get_live_ptfs(self, st: TickerState, date: pd.Timestamp) -> List[RollingPtf]:
        def _ptf_is_live(ptf: RollingPtf, today: pd.Timestamp) -> bool:
            if today < ptf.entry_date:
                return False
            if today <= ptf.exit_date:
                return True
            if ptf.meta.get("pending_close", False):
                return True
            if any(leg.qty != 0.0 for leg in ptf.legs):
                last_expiry = max(
                    (leg.expiry for leg in ptf.legs), default=ptf.exit_date
                )
                return today <= last_expiry
            return False

        return [ptf for ptf in st.rolling_ptfs.values() if _ptf_is_live(ptf, date)]

    def get_live_ptfs_between(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        rows: List[Dict[str, Any]] = []
        for ptf_id, meta in self.all_ptfs.items():
            if symbol is not None and meta["symbol"] != symbol:
                continue
            e1 = meta["entry_date"]
            e2 = meta["exit_date"]
            if (e1 <= end_date) and (e2 >= start_date):
                rows.append(meta.copy())
        return rows

    def _compute_vega_target(
        self, st: TickerState, date: pd.Timestamp, symbol: str
    ) -> float:
        return self.strategy.compute_vega_target(st)

    @staticmethod
    def _pnl_explain_leg(
        S_t, S_t1, sigma_t, sigma_t1, tau_t, tau_t1, r_t, r_t1,
        K, is_call, n_steps,
    ):
        """Full-repricing PnL attribution for one option leg.

        Subdivide the day into N steps.  At each step, start from state_i
        and reprice with exactly ONE factor bumped to its next-step value
        (all others frozen).  The price difference is that factor's
        contribution for the step.

        For step i, with base price V_i = BS(S_i, K, τ_i, r_i, σ_i):
          spot_pnl_i  = BS(S_{i+1}, K, τ_i, r_i, σ_i)  - V_i
          vega_pnl_i  = BS(S_i, K, τ_i, r_i, σ_{i+1})  - V_i
          theta_pnl_i = BS(S_i, K, τ_{i+1}, r_i, σ_i)  - V_i
          rho_pnl_i   = BS(S_i, K, τ_i, r_{i+1}, σ_i)  - V_i

        Spot PnL is reported as "gamma" (full repricing, no delta/gamma split).
        Residual = actual total PnL - sum of all buckets (cross-terms).

        Returns per-contract (not multiplied by qty) attribution dict.
        """
        zero = {"gamma": 0.0, "vega": 0.0, "theta": 0.0,
                "rho": 0.0, "residual": 0.0}

        # Validate inputs
        if not all(np.isfinite([S_t, S_t1, sigma_t, sigma_t1,
                                 tau_t, tau_t1, r_t, r_t1, K])):
            return zero
        if S_t <= 0 or sigma_t <= 0 or tau_t <= 0:
            return zero

        # Per-step increments
        dS = (S_t1 - S_t) / n_steps
        dsig = (sigma_t1 - sigma_t) / n_steps
        dtau = (tau_t1 - tau_t) / n_steps          # negative (time decays)
        dr = (r_t1 - r_t) / n_steps

        # Build N+1 state vectors
        t_idx = np.arange(n_steps + 1, dtype=float)
        S_path = S_t + t_idx * dS
        sig_path = sigma_t + t_idx * dsig
        tau_path = tau_t + t_idx * dtau
        r_path = r_t + t_idx * dr

        # Clamp to valid BS domain
        S_path = np.maximum(S_path, 1e-8)
        sig_path = np.maximum(sig_path, 1e-8)
        tau_path = np.maximum(tau_path, 1e-8)

        K_arr = np.full(n_steps, K)
        call_arr = np.full(n_steps, is_call)

        # Base price at each of the N starting nodes
        V_base = bs_price(S_path[:-1], K_arr, tau_path[:-1],
                          r_path[:-1], sig_path[:-1], call_arr)

        # --- Spot (reported as "gamma"): reprice with S bumped, everything else frozen ---
        V_spot = bs_price(S_path[1:], K_arr, tau_path[:-1],
                          r_path[:-1], sig_path[:-1], call_arr)
        gamma_attr = V_spot - V_base

        # --- Vega: reprice with σ bumped to σ_{i+1}, everything else frozen ---
        V_vol = bs_price(S_path[:-1], K_arr, tau_path[:-1],
                         r_path[:-1], sig_path[1:], call_arr)
        vega_attr = V_vol - V_base

        # --- Theta: reprice with τ bumped to τ_{i+1}, everything else frozen ---
        tau_next = np.maximum(tau_path[1:], 1e-8)
        V_time = bs_price(S_path[:-1], K_arr, tau_next,
                          r_path[:-1], sig_path[:-1], call_arr)
        theta_attr = V_time - V_base

        # --- Rho: reprice with r bumped to r_{i+1}, everything else frozen ---
        V_rate = bs_price(S_path[:-1], K_arr, tau_path[:-1],
                          r_path[1:], sig_path[:-1], call_arr)
        rho_attr = V_rate - V_base

        # Sum each bucket across all N steps
        sum_gamma = float(np.nansum(gamma_attr))
        sum_vega = float(np.nansum(vega_attr))
        sum_theta = float(np.nansum(theta_attr))
        sum_rho = float(np.nansum(rho_attr))

        # Total actual PnL (full repricing end-to-end)
        V_start = bs_price(
            np.array([S_t]), np.array([K]),
            np.array([max(tau_t, 1e-8)]), np.array([r_t]),
            np.array([sigma_t]), np.array([is_call]),
        )[0]
        V_end = bs_price(
            np.array([S_t1]), np.array([K]),
            np.array([max(tau_t1, 1e-8)]), np.array([r_t1]),
            np.array([sigma_t1]), np.array([is_call]),
        )[0]
        if not np.isfinite(V_start) or not np.isfinite(V_end):
            return zero
        total_pnl = float(V_end - V_start)

        # Residual = cross-terms (vanna, charm, veta, etc.)
        explained = sum_gamma + sum_vega + sum_theta + sum_rho
        residual = total_pnl - explained

        return {
            "gamma": sum_gamma,
            "vega": sum_vega,
            "theta": sum_theta,
            "rho": sum_rho,
            "residual": residual,
        }

    def process_symbol_date(self, symbol: str, date: pd.Timestamp) -> None:
        st = self.state[symbol]
        cfg = self.config
        perf_prev = st.perf

        # 0) Spot
        spot = self.market.get_spot(symbol, date)
        if spot is None or not np.isfinite(spot):
            return

        # PnL diagnostics
        option_pnl_entry = 0.0
        option_pnl_close = 0.0
        option_pnl_expire = 0.0
        pnl_gamma = 0.0
        pnl_vega = 0.0
        pnl_theta = 0.0
        pnl_rho = 0.0
        pnl_explain_residual = 0.0
        pnl_tc_opt = 0.0

        # 1) Corporate actions (splits)
        self._apply_split_to_ptfs(symbol, date)

        # 2) Compute vega target and let strategy create new PTFs
        vega_target = self._compute_vega_target(st, date, symbol)
        self.strategy.on_day(date, symbol, st, self.market, self, vega_target)

        # 3) Get live portfolios and today's chain
        live_ptfs = self._get_live_ptfs(st, date)

        chain_today = self.market.get_chain(symbol, date)
        if not chain_today.empty:
            chain_today["cid"] = chain_today["contractID"].astype(str)
            chain_idx = chain_today.set_index("cid")
        else:
            chain_idx = None

        # 3.5) Collect PnL Explain candidates (before section 4 modifies qty)
        _explain_legs = []
        if cfg.pnl_explain and st.last_spot is not None and st.last_date is not None:
            for ptf in live_ptfs:
                for leg in ptf.legs:
                    if leg.qty == 0.0 or leg.prev_iv <= 0.0:
                        continue
                    if leg.expiry <= date:
                        continue  # expiring today — payoff handled separately
                    cid = leg.contract_id
                    if chain_idx is None or cid not in chain_idx.index:
                        continue
                    row_ex = chain_idx.loc[cid]
                    iv_today = float(row_ex.get("implied_volatility", np.nan))
                    if not np.isfinite(iv_today) or iv_today <= 0.0:
                        continue
                    _explain_legs.append({
                        "qty": leg.qty,
                        "strike": leg.strike,
                        "expiry": leg.expiry,
                        "is_call": leg.opt_type == "C",
                        "prev_iv": leg.prev_iv,
                        "iv_today": iv_today,
                    })

        # 4) Entry / Close / Expiry cashflows
        for ptf in live_ptfs:
            for leg in ptf.legs:
                cid = leg.contract_id
                expiry = leg.expiry
                strike = leg.strike
                opt_type = leg.opt_type
                qty = leg.qty

                if qty == 0.0:
                    continue

                # ---- Expiry payoff ----
                if (expiry == date) and (ptf.entry_date <= date) and (leg.qty != 0.0):
                    if opt_type == "C":
                        intrinsic = max(spot - strike, 0.0)
                    else:
                        intrinsic = max(strike - spot, 0.0)

                    payoff = intrinsic * qty
                    st.cash += payoff
                    option_pnl_expire += payoff

                    leg.qty = 0.0
                    if ptf.meta.get("pending_close", False):
                        ptf.meta["pending_close"] = False
                    continue

                pending_close = ptf.meta.get("pending_close", False)
                pending_since = ptf.meta.get("pending_close_since")

                if chain_idx is None or cid not in chain_idx.index:
                    if pending_close:
                        max_wait = cfg.exit_fallback_max_wait_bdays
                        if pending_since is not None:
                            bdays = pd.bdate_range(pending_since, date)
                            wait = max(0, len(bdays) - 1)
                        else:
                            wait = max_wait + 1

                        if (wait >= max_wait) and (leg.qty != 0.0):
                            if opt_type == "C":
                                intrinsic = max(spot - strike, 0.0)
                            else:
                                intrinsic = max(strike - spot, 0.0)
                            cash_change = qty * intrinsic
                            st.cash += cash_change
                            option_pnl_close += -cash_change
                            leg.qty = 0.0
                            ptf.meta["pending_close"] = False
                    elif (date == ptf.exit_date) and (expiry > date) and (qty != 0.0):
                        if cfg.exit_fallback_mode == "next":
                            ptf.meta["pending_close"] = True
                            ptf.meta["pending_close_since"] = date
                        else:
                            if opt_type == "C":
                                intrinsic = max(spot - strike, 0.0)
                            else:
                                intrinsic = max(strike - spot, 0.0)
                            cash_change = qty * intrinsic
                            st.cash += cash_change
                            option_pnl_close += -cash_change
                            leg.qty = 0.0
                    continue

                row_opt = chain_idx.loc[cid]
                mid_price = float(row_opt["mid"])

                # ---- Entry day ----
                if date == ptf.entry_date:
                    trade_qty = qty
                    trade_price = self.execution.get_option_trade_price(
                        row_opt, trade_qty, mid_price
                    )
                    book = self.execution.book_option_trade(
                        st, trade_qty, trade_price, mid_price
                    )
                    pnl_tc_opt += book["tc_spread"] + book["tc_comm"]
                    option_pnl_entry += -book["cash_change"]

                # ---- Pending close ----
                if pending_close and (expiry > date) and (qty != 0.0):
                    trade_qty = -qty
                    trade_price = self.execution.get_option_trade_price(
                        row_opt, trade_qty, mid_price
                    )
                    book = self.execution.book_option_trade(
                        st, trade_qty, trade_price, mid_price
                    )
                    pnl_tc_opt += book["tc_spread"] + book["tc_comm"]
                    option_pnl_close += -book["cash_change"]
                    leg.qty = 0.0
                    ptf.meta["pending_close"] = False
                    continue

                # ---- Exit day (non-expiry) ----
                if (date == ptf.exit_date) and (expiry > date) and (qty != 0.0):
                    trade_qty = -qty
                    trade_price = self.execution.get_option_trade_price(
                        row_opt, trade_qty, mid_price
                    )
                    book = self.execution.book_option_trade(
                        st, trade_qty, trade_price, mid_price
                    )
                    pnl_tc_opt += book["tc_spread"] + book["tc_comm"]
                    option_pnl_close += -book["cash_change"]
                    leg.qty = 0.0
                    continue

        # 4.5) PnL Explain — pathwise Euler decomposition
        if cfg.pnl_explain and _explain_legs:
            r_t = self.market.get_rate(st.last_date)
            r_t1 = self.market.get_rate(date)
            n_steps = cfg.pnl_explain_n_steps
            for info in _explain_legs:
                S_t = st.last_spot
                S_t1 = spot
                sigma_t = info["prev_iv"]
                sigma_t1 = info["iv_today"]
                tau_t = max((info["expiry"] - st.last_date).days, 1) / 365.0
                tau_t1 = max((info["expiry"] - date).days, 1) / 365.0
                K = info["strike"]
                is_call = info["is_call"]
                qty_ex = info["qty"]

                res = self._pnl_explain_leg(
                    S_t, S_t1, sigma_t, sigma_t1,
                    tau_t, tau_t1, r_t, r_t1, K, is_call, n_steps,
                )
                pnl_gamma += qty_ex * res["gamma"]
                pnl_vega += qty_ex * res["vega"]
                pnl_theta += qty_ex * res["theta"]
                pnl_rho += qty_ex * res["rho"]
                pnl_explain_residual += qty_ex * res["residual"]

        # 5) Option MtM and greek exposures
        mtm_options = 0.0
        port_delta_today = 0.0
        port_gamma_today = 0.0
        port_vega_today = 0.0
        port_theta_today = 0.0

        for ptf in live_ptfs:
            for leg_idx, leg in enumerate(ptf.legs):
                cid = leg.contract_id
                qty = leg.qty

                if qty == 0.0:
                    continue

                if chain_idx is None or cid not in chain_idx.index:
                    continue

                row_opt = chain_idx.loc[cid]
                mid_price = float(row_opt["mid"])
                iv = float(row_opt.get("implied_volatility", np.nan))
                delta = float(row_opt.get("delta", 0.0))
                gamma = float(row_opt.get("gamma", 0.0))
                vega = float(row_opt.get("vega", 0.0))
                theta = float(row_opt.get("theta", 0.0))
                # Guard NaN Greeks -> 0 (prevents NaN contaminating
                # portfolio delta → delta hedge → cash → perf)
                if not np.isfinite(delta):
                    delta = 0.0
                if not np.isfinite(gamma):
                    gamma = 0.0
                if not np.isfinite(vega):
                    vega = 0.0
                if not np.isfinite(theta):
                    theta = 0.0

                bid = float(row_opt.get("bid", np.nan))
                ask = float(row_opt.get("ask", np.nan))
                log_mode = cfg.trade_log_mode
                log_this = (
                    (log_mode == "all")
                    or (log_mode == "entries" and date == ptf.entry_date)
                    or (log_mode == "light" and date == ptf.entry_date
                        and leg_idx < 5)
                )
                if log_this:
                    self.trade_rows.append({
                        "Date": date,
                        "Trade Date": ptf.entry_date,
                        "Exit Date": ptf.exit_date,
                        "Symbol": symbol,
                        "ContractID": cid,
                        "Expiry": leg.expiry,
                        "Strike": leg.strike,
                        "Type": leg.opt_type,
                        "TradeQty": qty,
                        "Mid": mid_price,
                        "Bid": bid,
                        "Ask": ask,
                        "TradePrice": mid_price,
                        "TradeNotional": qty * mid_price,
                        "Spot": spot,
                        "IV": iv,
                        "Delta": delta,
                        "Gamma": gamma,
                        "Vega": vega,
                        "Theta": theta,
                    })

                mtm_options += qty * mid_price
                port_delta_today += qty * delta
                port_gamma_today += qty * gamma
                port_vega_today += qty * vega
                port_theta_today += qty * theta

                # Taylor-based PnL attribution (fallback when pnl_explain is off)
                if not cfg.pnl_explain:
                    dS = 0.0 if st.last_spot is None else (spot - st.last_spot)
                    pnl_gamma += leg.prev_delta * dS * qty
                    pnl_gamma += 0.5 * leg.prev_gamma * (dS ** 2) * qty
                    if np.isfinite(iv) and np.isfinite(leg.prev_iv):
                        dIV = iv - leg.prev_iv
                        pnl_vega += leg.prev_vega * dIV * qty
                    pnl_theta += leg.prev_theta * (1.0 / 252.0) * qty

                leg.prev_price = mid_price
                leg.prev_iv = iv if np.isfinite(iv) else leg.prev_iv
                leg.prev_delta = delta
                leg.prev_gamma = gamma
                leg.prev_vega = vega
                leg.prev_theta = theta

        st.mtm_options = mtm_options

        # 5b) Strategy hook: adjust hedge delta (e.g. skew delta correction)
        hedge_delta_adj = -self.strategy.adjust_hedge_delta(
            live_ptfs, chain_idx, spot, port_delta_today,
        )
        port_delta_for_hedge = port_delta_today + hedge_delta_adj

        # 6) Stock hedge & MTM
        hedge_res = self.execution.apply_delta_hedge(
            st, port_delta_for_hedge, spot
        )
        pnl_delta_hedge = hedge_res["pnl_delta_hedge"]
        pnl_tc_stock = hedge_res["pnl_tc_stock"]

        if not np.isfinite(st.stock_pos_close):
            st.stock_pos_close = 0.0
        st.mtm_stock = st.stock_pos_close * spot

        # 7) Perf & PnL
        st.perf = st.cash + st.mtm_options + st.mtm_stock
        pnl_total = st.perf - perf_prev

        # TC attribution: stock TC into gamma (spot/hedge bucket), option TC into vega
        pnl_gamma -= pnl_tc_stock
        pnl_vega -= pnl_tc_opt

        # Residual = total PnL minus all explained greek components
        pnl_residual = pnl_total - (
            pnl_gamma + pnl_vega + pnl_theta + pnl_rho
            + pnl_explain_residual
        )

        st.cum_pnl += pnl_total
        st.cum_pnl_gamma += pnl_gamma
        st.cum_pnl_vega += pnl_vega
        st.cum_pnl_theta += pnl_theta
        st.cum_pnl_rho += pnl_rho
        st.cum_pnl_delta_hedge += pnl_delta_hedge
        st.cum_pnl_tc += pnl_tc_opt + pnl_tc_stock
        st.cum_pnl_residual += pnl_residual

        st.last_spot = spot
        st.last_date = date

        # 8) Daily row
        self.daily_pnl_rows.append({
            "Date": date,
            "Symbol": symbol,
            "Perf": st.perf,
            "Cash": st.cash,
            "MTM_Options": st.mtm_options,
            "MTM_Stock": st.mtm_stock,
            "Spot": spot,
            "DailyPnL": pnl_total,
            "OptionPnL_Entry": option_pnl_entry,
            "OptionPnL_Close": option_pnl_close,
            "OptionPnL_Expire": option_pnl_expire,
            "CumPnL": st.cum_pnl,
            "CumPnL_gamma": st.cum_pnl_gamma,
            "CumPnL_vega": st.cum_pnl_vega,
            "CumPnL_theta": st.cum_pnl_theta,
            "CumPnL_rho": st.cum_pnl_rho,
            "CumPnL_residual": st.cum_pnl_residual,
            "PnL_deltaHedge": pnl_delta_hedge,
            "TC_options": pnl_tc_opt,
            "TC_stock": pnl_tc_stock,
            "CumPnL_deltaHedge": st.cum_pnl_delta_hedge,
            "CumPnL_TC": st.cum_pnl_tc,
            "Delta": port_delta_for_hedge,
            "BS_Delta": port_delta_today,
            "Skew_Delta": hedge_delta_adj,
            "Gamma": port_gamma_today,
            "Vega": port_vega_today,
            "Theta": port_theta_today,
            "StockPos": st.stock_pos_close,
        })
