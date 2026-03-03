"""
HAR-RV: Heterogeneous Autoregressive Realized Volatility Forecasting Model.

Predicts h-day forward realized volatility using:
    RV_forecast(t+h) = alpha + beta_d * RV_1d(t) + beta_w * RV_5d(t) + beta_m * RV_22d(t)

Based on Corsi (2009). Expanding-window OLS with periodic refit (no look-ahead).

Usage:
    model = HARRVModel(min_history=252, horizon=5)
    har_df = model.fit_predict(spot_series, symbol="QQQ")
    # har_df has columns: rv_1d, rv_5d, rv_22d, har_rv_forecast
"""

import numpy as np
import pandas as pd

from base_signals import compute_rolling_zscore


class HARRVModel:
    """Heterogeneous Autoregressive Realized Volatility model."""

    def __init__(self, min_history: int = 252, horizon: int = 5, refit_freq: int = 20):
        """
        Parameters
        ----------
        min_history : int
            Minimum days of history before first forecast (default 252 = 1 year).
        horizon : int
            Forecast horizon in trading days (default 5 = 1 week).
        refit_freq : int
            Refit OLS coefficients every N business days (default 20).
        """
        self.min_history = min_history
        self.horizon = horizon
        self.refit_freq = refit_freq
        self._cache = {}

    def fit_predict(self, spot_series: pd.Series, symbol: str = "QQQ") -> pd.DataFrame:
        """
        Build expanding-window HAR-RV forecasts for the entire series.

        Parameters
        ----------
        spot_series : pd.Series
            Daily spot prices (index = DatetimeIndex).
        symbol : str
            Cache key.

        Returns
        -------
        pd.DataFrame with columns:
            rv_1d           - 1-day annualized realized variance
            rv_5d           - 5-day (weekly) average RV
            rv_22d          - 22-day (monthly) average RV
            har_rv_forecast - predicted h-day forward RV (annualized variance)
        """
        if symbol in self._cache:
            return self._cache[symbol]

        idx = spot_series.index
        n = len(idx)

        # Compute realized variance components
        log_ret = np.log(spot_series / spot_series.shift(1))
        rv_1d = log_ret ** 2 * 252  # annualized daily variance
        rv_5d = rv_1d.rolling(5, min_periods=3).mean()
        rv_22d = rv_1d.rolling(22, min_periods=10).mean()

        # Forward RV target: average rv_1d over next h days (for training only)
        rv_fwd = rv_1d.rolling(self.horizon, min_periods=self.horizon).mean().shift(-self.horizon)

        # Feature matrix
        features = pd.DataFrame({
            "rv_1d": rv_1d,
            "rv_5d": rv_5d,
            "rv_22d": rv_22d,
        }, index=idx)

        # Expanding-window OLS with periodic refit
        forecasts = np.full(n, np.nan)
        coeffs = None
        last_fit = -self.refit_freq  # force fit on first eligible date

        for t in range(self.min_history, n):
            # Refit every refit_freq days
            if t - last_fit >= self.refit_freq or coeffs is None:
                train_end = t - self.horizon  # no look-ahead
                if train_end < self.min_history // 2:
                    continue

                X_train = features.iloc[:train_end + 1].dropna()
                y_train = rv_fwd.reindex(X_train.index).dropna()
                X_train = X_train.reindex(y_train.index)

                if len(X_train) < 60:
                    continue

                # OLS: y = X @ beta + intercept
                X_arr = np.column_stack([np.ones(len(X_train)), X_train.values])
                y_arr = y_train.values

                try:
                    coeffs = np.linalg.lstsq(X_arr, y_arr, rcond=None)[0]
                    last_fit = t
                except np.linalg.LinAlgError:
                    continue

            # Predict at time t
            if coeffs is not None:
                x_t = features.iloc[t]
                if x_t.isna().any():
                    continue
                x_pred = np.array([1.0, x_t["rv_1d"], x_t["rv_5d"], x_t["rv_22d"]])
                pred = np.dot(coeffs, x_pred)
                forecasts[t] = max(pred, 0.0)  # variance can't be negative

        result = features.copy()
        result["har_rv_forecast"] = forecasts

        self._cache[symbol] = result
        print(f"[HAR-RV] {symbol}: {np.isfinite(forecasts).sum()} forecasts "
              f"({self.min_history}+ history, {self.horizon}D horizon, "
              f"refit every {self.refit_freq}D)")

        return result


def add_har_vrp_signals(har_df: pd.DataFrame, atm_iv_series: pd.Series) -> pd.DataFrame:
    """
    Add VRP signals derived from HAR-RV forecast vs ATM IV.

    Parameters
    ----------
    har_df : pd.DataFrame
        Output of HARRVModel.fit_predict().
    atm_iv_series : pd.Series
        ATM implied volatility (as decimal, e.g. 0.20 for 20%).
        Must be aligned to har_df index.

    Returns
    -------
    har_df with additional columns:
        har_vrp_signal  - ATM_IV^2 - har_rv_forecast (pos = sell, neg = buy gamma)
        har_vrp_zscore  - 60-day rolling z-score of har_vrp_signal
    """
    iv_aligned = atm_iv_series.reindex(har_df.index, method="ffill")
    har_df["har_vrp_signal"] = iv_aligned ** 2 - har_df["har_rv_forecast"]
    har_df["har_vrp_zscore"] = compute_rolling_zscore(har_df["har_vrp_signal"], 60)
    return har_df
