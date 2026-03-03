"""
Flagship Signal Engine -- production signal computation.

Extends BaseSignalEngine with validated signals:
  - Composite Regime Score (momentum + credit + vol + term structure)
  - Credit-Equity Divergence
  - RV Surprise (Parkinson / close-to-close)
  - Return Autocorrelation, Realized Skewness

V3 additions:
  - Vol surface integration (SVI-fitted features from cache)
  - _signal_scale() for continuous position sizing [0.0, 2.0]
  - _vol_sig() for reading SVI vol surface features

V4 additions:
  - HAR-RV forecasting integration (har_rv.py)
  - _har_sig() for reading HAR-RV signal values

Chain-derived signals (GEX, BF25, skew, IV percentile, VRP TS) are computed
on-the-fly in strategies via helper functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from base_signals import BaseSignalEngine, compute_rolling_zscore
from har_rv import HARRVModel, add_har_vrp_signals


# ================================================================
# Vol Surface Cache Loader
# ================================================================

_VOL_SURFACE_CACHE = {}  # symbol -> DataFrame


def _load_vol_surface(symbol: str = "QQQ") -> pd.DataFrame:
    """Load SVI vol surface features from cache parquet."""
    if symbol in _VOL_SURFACE_CACHE:
        return _VOL_SURFACE_CACHE[symbol]
    path = Path(f"cache_options/vol_surface_{symbol}.parquet")
    if not path.exists():
        _VOL_SURFACE_CACHE[symbol] = pd.DataFrame()
        return _VOL_SURFACE_CACHE[symbol]
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).normalize()
    _VOL_SURFACE_CACHE[symbol] = df
    return df


# ================================================================
# Extended macro signal engine
# ================================================================

class FlagshipSignalEngine(BaseSignalEngine):
    """Extends BaseSignalEngine with validated cross-asset signals."""

    def build_signal_df(self, spot_series: pd.Series,
                        ohlc_df: pd.DataFrame = None) -> pd.DataFrame:
        """Build full signal DataFrame."""
        sig = super().build_signal_df(spot_series)

        # -- Credit-Equity Divergence --
        if "hy_oas_zscore" in sig.columns:
            ret_20d_z = compute_rolling_zscore(spot_series.pct_change(20), 60)
            sig["credit_equity_div"] = sig["hy_oas_zscore"] - (-ret_20d_z)

        # -- VVIX/VIX Ratio --
        if "vvix" in sig.columns and "vix_cboe" in sig.columns:
            sig["vvix_vix_ratio"] = sig["vvix"] / sig["vix_cboe"].replace(0, np.nan)

        # -- RV Surprise (Parkinson / close-to-close) --
        if ohlc_df is not None and all(c in ohlc_df.columns for c in ["high", "low"]):
            log_hl = np.log(ohlc_df["high"] / ohlc_df["low"])
            parkinson_5d = log_hl.rolling(5, min_periods=3).apply(
                lambda x: np.sqrt((1 / (4 * np.log(2))) * (x ** 2).mean()) * np.sqrt(252),
                raw=True,
            )
            rv5 = sig.get("rv_5d")
            if rv5 is not None:
                sig["rv_surprise"] = parkinson_5d / rv5.replace(0, np.nan)

        # -- Return Autocorrelation (20-day rolling) --
        daily_ret = spot_series.pct_change()
        sig["ret_autocorr"] = daily_ret.rolling(20, min_periods=10).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 5 else np.nan,
            raw=False,
        )

        # -- Realized Skewness & Kurtosis --
        sig["realized_skew"] = daily_ret.rolling(21, min_periods=10).skew()
        sig["realized_kurt"] = daily_ret.rolling(21, min_periods=10).kurt()

        # -- Composite Regime Score --
        sig["regime_score"] = self._compute_regime_score(sig, spot_series)

        # -- HAR-RV Forecasting (V4) --
        try:
            har_model = HARRVModel(min_history=252, horizon=5, refit_freq=20)
            har_df = har_model.fit_predict(spot_series)
            if not har_df.empty and "har_rv_forecast" in har_df.columns:
                sig["har_rv_forecast"] = har_df["har_rv_forecast"]
                # Use VIX as IV proxy for VRP signal; strategies override with SVI ATM IV
                if "vix" in sig.columns:
                    atm_iv_proxy = sig["vix"] / 100.0
                    sig["har_vrp_signal"] = atm_iv_proxy ** 2 - har_df["har_rv_forecast"]
                    sig["har_vrp_zscore"] = compute_rolling_zscore(sig["har_vrp_signal"], 60)
        except Exception as e:
            print(f"[SIG] HAR-RV failed: {e}")

        return sig

    @staticmethod
    def _compute_regime_score(sig: pd.DataFrame, spot_series: pd.Series) -> pd.Series:
        """Composite regime score: [-1 (risk-off), +1 (risk-on)]."""
        sma = sig.get("sma_200")
        if sma is None:
            sma = spot_series.rolling(200, min_periods=100).mean()

        above_sma = spot_series > sma
        distance = np.abs(spot_series / sma - 1) / 0.05
        momentum_score = np.where(above_sma, np.clip(distance, 0, 1),
                                  -np.clip(distance, 0, 1))

        hy_z = sig.get("hy_oas_zscore", pd.Series(0.0, index=sig.index)).fillna(0)
        credit_score = -np.clip(hy_z.values, -2, 2) / 2

        vix_z = sig.get("vix_zscore_long", pd.Series(0.0, index=sig.index)).fillna(0)
        vol_score = -np.clip(vix_z.values, -2, 2) / 2

        ts = sig.get("vix_ts_ratio", pd.Series(1.0, index=sig.index)).fillna(1)
        ts_score = np.clip((1.0 - ts.values) * 10, -1, 1)

        return pd.Series(
            0.30 * momentum_score + 0.25 * credit_score + 0.25 * vol_score + 0.20 * ts_score,
            index=sig.index,
        )


# ================================================================
# Chain-derived signal helpers (call from on_day with daily chain)
# ================================================================

def compute_gex(chain: pd.DataFrame, spot: float) -> float:
    """Directional GEX: sum(gamma * OI * spot * sign) / (spot * 1e6)."""
    if chain is None or chain.empty:
        return np.nan
    if "gamma" not in chain.columns or "open_interest" not in chain.columns:
        return np.nan
    gamma = chain["gamma"].astype(float).abs()
    oi = chain["open_interest"].astype(float).fillna(0)
    sign = np.where(chain["type"] == "C", 1.0, -1.0)
    gex = (gamma * oi * spot * sign * 100).sum()
    return gex / (spot * 1e6)


def compute_bf25(chain: pd.DataFrame) -> float:
    """Butterfly spread: 0.5*(IV_25D_call + IV_25D_put) - IV_ATM."""
    if chain is None or chain.empty:
        return np.nan
    iv = chain["implied_volatility"].astype(float)
    delta = chain["delta"].astype(float)
    dte_mask = chain["dte"].between(10, 35) if "dte" in chain.columns else pd.Series(True, index=chain.index)
    sub = chain[dte_mask]
    if sub.empty:
        return np.nan
    iv_s, delta_s = iv[dte_mask], delta[dte_mask]
    atm = iv_s[(sub["type"] == "C") & delta_s.between(0.40, 0.60)]
    atm_iv = float(atm.median()) if not atm.empty else np.nan
    c25 = iv_s[(sub["type"] == "C") & delta_s.between(0.20, 0.30)]
    c25_iv = float(c25.median()) if not c25.empty else np.nan
    p25 = iv_s[(sub["type"] == "P") & delta_s.between(-0.30, -0.20)]
    p25_iv = float(p25.median()) if not p25.empty else np.nan
    if all(np.isfinite([atm_iv, c25_iv, p25_iv])):
        return 0.5 * (c25_iv + p25_iv) - atm_iv
    return np.nan


def compute_chain_skew(chain: pd.DataFrame) -> float:
    """25D put IV / ATM IV. Values > 1 = puts expensive."""
    if chain is None or chain.empty:
        return np.nan
    sub = chain[chain["dte"].between(10, 40)] if "dte" in chain.columns else chain
    if sub.empty:
        sub = chain[chain["dte"].between(5, 45)] if "dte" in chain.columns else chain
    if sub.empty:
        return np.nan
    iv = sub["implied_volatility"].astype(float)
    delta = sub["delta"].astype(float)
    atm = iv[(sub["type"] == "C") & delta.between(0.40, 0.60)]
    atm_iv = float(atm.median()) if not atm.empty else np.nan
    put25 = iv[(sub["type"] == "P") & delta.between(-0.30, -0.20)]
    put_iv = float(put25.median()) if not put25.empty else np.nan
    if np.isfinite(atm_iv) and np.isfinite(put_iv) and atm_iv > 0:
        return put_iv / atm_iv
    return np.nan


def compute_atm_iv(chain: pd.DataFrame, spot: float, target_dte: int = 30) -> float:
    """Get ATM IV for a given target DTE range."""
    sub = chain[(chain["dte"].between(target_dte - 5, target_dte + 10)) &
                 (chain["type"] == "C")] if "dte" in chain.columns else chain[chain["type"] == "C"]
    if sub.empty:
        sub = chain[(chain["dte"].between(5, 45)) & (chain["type"] == "C")] if "dte" in chain.columns else chain[chain["type"] == "C"]
    if sub.empty:
        return np.nan
    if "delta" in sub.columns:
        delta_diff = (sub["delta"].astype(float) - 0.50).abs()
        atm = sub.loc[delta_diff.nsmallest(3).index]
    else:
        mny = sub["strike"] / spot
        atm = sub.loc[(mny - 1.0).abs().nsmallest(3).index]
    return float(atm["implied_volatility"].median())


# ================================================================
# Signal mixin for flagship strategies (V3: with vol surface + continuous sizing)
# ================================================================

class FlagshipSignalMixin:
    """Signal infrastructure for flagship strategies.

    V3 additions:
    - _vol_sig(): read SVI vol surface features
    - _signal_scale(): combine signals → continuous scale [0.1, 2.0]
    """

    def _build_flagship_signals(self, market) -> dict:
        """Build per-symbol signal DataFrames using FlagshipSignalEngine."""
        engine = FlagshipSignalEngine()
        signals = {}
        for sym in self.config.symbols:
            spot_df = market.spot.get(sym)
            if spot_df is None or spot_df.empty:
                continue
            ohlc = None
            if all(c in spot_df.columns for c in ["open", "high", "low", "close"]):
                ohlc = spot_df
            sig_df = engine.build_signal_df(spot_df["spot"], ohlc_df=ohlc)

            # V4: Override HAR VRP with SVI ATM IV (more accurate than VIX proxy)
            vol_surf = _load_vol_surface(sym)
            if (not vol_surf.empty and "svi_30d_atm_iv" in vol_surf.columns
                    and "har_rv_forecast" in sig_df.columns):
                atm_iv = vol_surf["svi_30d_atm_iv"].reindex(sig_df.index, method="ffill")
                valid = atm_iv.notna() & sig_df["har_rv_forecast"].notna()
                if valid.any():
                    sig_df.loc[valid, "har_vrp_signal"] = (
                        atm_iv[valid] ** 2 - sig_df.loc[valid, "har_rv_forecast"]
                    )
                    sig_df["har_vrp_zscore"] = compute_rolling_zscore(
                        sig_df["har_vrp_signal"], 60
                    )

            signals[sym] = sig_df
        return signals

    def _load_vol_surface(self):
        """Load vol surface cache. Call from initialize()."""
        self._vol_surface = {}
        for sym in self.config.symbols:
            df = _load_vol_surface(sym)
            if not df.empty:
                self._vol_surface[sym] = df

    def _har_sig(self, symbol, date, col, default=np.nan):
        """Read HAR-RV signal value (stored in signals df)."""
        return self._sig(symbol, date, col, default)

    def _sig(self, symbol, date, col, default=np.nan):
        """Safely read one signal value."""
        sig_df = getattr(self, "signals", {}).get(symbol)
        if sig_df is None or date not in sig_df.index:
            return default
        if col not in sig_df.columns:
            return default
        val = sig_df.at[date, col]
        return val if np.isfinite(val) else default

    def _vol_sig(self, symbol, date, col, default=np.nan):
        """Read one SVI vol surface feature value."""
        vs = getattr(self, "_vol_surface", {}).get(symbol)
        if vs is None or vs.empty:
            return default
        if date not in vs.index:
            # Try nearest prior date
            prior = vs.index[vs.index <= date]
            if prior.empty:
                return default
            date = prior[-1]
        if col not in vs.columns:
            return default
        val = vs.at[date, col]
        return float(val) if np.isfinite(val) else default

    def _regime(self, symbol, date) -> float:
        """Get regime score [-1, +1]."""
        return self._sig(symbol, date, "regime_score", 0.0)

    def _iv_pctl(self, atm_iv: float, symbol: str) -> float:
        """Compute IV percentile from running history."""
        hist = self._iv_history.setdefault(symbol, [])
        if np.isfinite(atm_iv):
            hist.append(atm_iv)
        if len(hist) < 60:
            return np.nan
        window = hist[-252:]
        arr = np.array(window)
        return float(np.sum(arr <= atm_iv) / len(arr))

    def _signal_scale(self, symbol, date, signal_config: dict) -> float:
        """
        Combine multiple signals into a continuous scale in [0.0, 2.0].

        signal_config: dict mapping signal_name -> (transform_fn, weight)
            transform_fn: callable(raw_value) -> score in [-1, +1]
            weight: relative weight (will be normalized)

        Returns float in [0.0, 2.0]:
            1.0 = neutral (signals average to 0)
            2.0 = maximum conviction (all signals at +1)
            0.0 = no trade (all signals at -1)
        """
        scores = []
        weights = []
        for sig_name, (transform, weight) in signal_config.items():
            # Try vol surface first, then standard signals
            raw = self._vol_sig(symbol, date, sig_name, np.nan)
            if not np.isfinite(raw):
                raw = self._sig(symbol, date, sig_name, 0.0)
            score = np.clip(transform(raw), -1.0, 1.0)
            scores.append(score)
            weights.append(weight)

        if not scores:
            return 1.0

        w = np.array(weights, dtype=float)
        w = w / w.sum()
        composite = float(np.dot(scores, w))

        # Map [-1, +1] → [0.0, 2.0]: scale = 1.0 + composite
        return float(np.clip(1.0 + composite, 0.0, 2.0))
