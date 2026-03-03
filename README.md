# Systematic Volatility Strategies & Earnings IV Research

A quantitative research framework that (1) studies the behavior of implied volatility around earnings announcements across ~350 US equities and (2) implements a production portfolio of 8 systematic option strategies, backtested on both ETFs and indices with a custom-built options engine featuring Greek-level P&L attribution.

**Data**: ~350 single-name option chains (S&P 500 constituents, Alpha Vantage) for the IV study, plus 4 index/ETF underlyings for strategy backtesting across two institutional-grade sources:

| Source | Underlyings | Period | Scale |
|--------|-------------|--------|-------|
| Alpha Vantage | ~350 single names + QQQ, SPY | 2020-2025 | ~3.8M rows (QQQ) |
| OptionMetrics / WRDS | NDX, SPX | 2016-2025 | ~28M (NDX), ~40M (SPX) |

---

## Repository Structure

```
├── iv_stat_analysis.py           # Earnings IV study (Part 1)
├── signal_generator.py           #   Rule-based signal construction
├── signal_generator_ml.py        #   Walk-forward GBM signals + SHAP
├── bs_pricing.py                 #   Black-Scholes pricer, Greeks, IV solver
│
├── flagship_final.py             # Strategy suite runner -- ETFs (QQQ/SPY)
├── flagship_final_cboe.py        # Strategy suite runner -- Indices (NDX/SPX via OptionMetrics)
├── flagship_signals.py           #   Signal engine (VIX, HAR-RV, regime, vol surface)
├── flagship_metrics.py           #   Composite portfolio analytics + quality gates
├── base_signals.py               #   Base signal engine (VIX, VVIX, CBOE, FRED, macro)
├── har_rv.py                     #   HAR-RV realized vol forecasting model
├── build_vol_surface.py          #   SVI vol surface fitting per date x expiry
│
├── backtester/                   # Options backtesting engine (built from scratch)
│   ├── config.py                 #   BacktestConfig dataclass
│   ├── models.py                 #   OptionLeg, RollingPtf, TickerState
│   ├── data_store.py             #   Two-tier cached data loading (predicate pushdown)
│   ├── engine.py                 #   Day-by-day orchestrator + strategy registry
│   ├── portfolio.py              #   Position lifecycle, MTM, splits, delta hedge
│   ├── execution.py              #   Trade pricing, spread model, transaction costs
│   ├── metrics.py                #   Risk-adjusted metrics + Greek P&L attribution
│   ├── reporting.py              #   Multi-sheet Excel export
│   └── strategies/
│       ├── base.py               #   Abstract Strategy interface
│       ├── earnings.py           #   Earnings straddle/strangle
│       ├── rolling.py            #   Calendar rolling (put-write, call-overwrite)
│       ├── variance_swap.py      #   Discrete variance swap replication
│       ├── calendar_spread.py    #   Calendar spreads
│       └── dispersion.py         #   Index vs single-name dispersion
│
├── data_pulling/                  # Data acquisition & enrichment
│   ├── alpha_vantage_option_pull.py    # EOD option chain download
│   ├── alpha_vantage_pull_earnings.py  # Earnings calendar download
│   ├── alpha_vantage_intraday_pull.py  # Intraday spot data download
│   ├── build_corporate_actions.py      # Split/dividend adjustments
│   └── enrich_options_bs.py            # Greeks enrichment via BS model
│
├── run_backtest.py               # Single-strategy runner
│
└── outputs/
    ├── flagship_final/           # ETF results (QQQ + SPY)
    ├── flagship_suite_cboe/      # Index results (NDX + SPX via OptionMetrics)
    └── earnings_iv_analysis.xlsx # Full IV study results
```

---

## Part 1 -- Systematic Strategy Suite

### The 8-Strategy Portfolio

**`flagship_final.py`** (ETFs) and **`flagship_final_cboe.py`** (indices) implement 8 strategies allocated 60% carry / 40% hedge with equal weight within each bucket. The carry side harvests the variance risk premium through short options structures; the hedge side provides convex protection during stress periods.


Hedge strategies use constant sizing (not performance-scaled) and backfill positions from Jan 2019 to avoid cold-start bias on monthly/long-hold positions.

Strategies are backtested on all 4 underlyings via two runners: `flagship_final.py` (QQQ/SPY, Alpha Vantage) and `flagship_final_cboe.py` (NDX/SPX, OptionMetrics). The CBOE runner adapts OptionMetrics conventions (strike/1000, annualized theta, per-unit vega) to the backtester schema.

Running the full suite (7 strategies x 4 underlyings = 28 backtests) completes in under an hour.

### Results

2+ Sharpe for the composite portfolio. See folder **`outputs/`** for full details.

**Carry (Sharpe ratios)**

| Strat | QQQ | SPY | NDX | SPX |
|-------|:---:|:---:|:---:|:---:|
| VXDH | 1.41 | 1.34 | 0.90 | 1.37 |
| OMDH | 1.53 | 1.37 | 0.34 | 1.10 |
| SDPS | 0.81 | 0.70 | 0.56 | 0.35 |
| VSS3 | 0.75 | 0.65 | -0.36 | 0.49 |

**Hedge (Sharpe ratios)**

| Strat | QQQ | SPY | NDX | SPX |
|-------|:---:|:---:|:---:|:---:|
| DVAS | 0.76 | 0.70 | 0.58 | 0.42 |
| XHGE | 0.39 | 0.74 | 0.51 | 0.57 |
| THT2 | 0.10 | 0.28 | -0.06 | 0.27 |

---

## The Backtesting Engine

**`backtester/`** is a modular, event-driven engine built from scratch for multi-leg options strategies. It processes EOD option chains day-by-day and manages positions through their full lifecycle.

### Architecture

```
BacktestEngine
 |
 |  for each trading day:
 |
 +-- Strategy.on_day()             --- decides entries, calls register_new_ptf()
 +-- PortfolioEngine               --- cashflows, expiry payoffs, early exits
 +-- ExecutionModel                --- trade pricing (mid / half-spread / bid-ask)
 +-- PortfolioEngine.mtm()         --- mark-to-market, Greeks, Taylor P&L attribution
 +-- Strategy.adjust_hedge_delta() --- optional skew correction hook
 +-- ExecutionModel.delta_hedge()  --- stock trade to flatten portfolio delta
 +-- emit daily row (~35 cols)     --- perf, Greeks, P&L by component
 |
 BacktestMetrics --- Sharpe, Sortino, Calmar, CVaR, Lo (2002) t-stat
 export_results() -- multi-sheet Excel workbook
```

### Daily Processing (8 stages per symbol)

| # | Stage | Detail |
|---|-------|--------|
| 0 | Spot | Fetch today's spot. Skip day if missing. |
| 1 | Corporate actions | Split-adjust strikes (/ factor), quantities (* factor), stock position. Zero stale Greeks. |
| 2 | Strategy | Call `strategy.on_day()`. Strategy opens positions via `register_new_ptf()`. |
| 3 | Live filter | Select portfolios where entry <= today <= exit. Index chain by contractID. |
| 4 | Cashflows | Expiry payoffs (intrinsic), entry bookings, exit closings, pending-close fallback. |
| 5 | MTM + Greeks | Mark every live leg. Taylor P&L: delta*dS, 0.5*gamma*dS^2, vega*dIV, theta/252. |
| 6 | Delta hedge | Flatten adjusted delta via stock. Hedge P&L = prev_stock_pos * dS. |
| 7 | Performance | perf = cash + MTM_options + MTM_stock. Attribute TC to Greek buckets. |

---


## Part 2 -- Implied Volatility Around Earnings

### Research Question

> Is there a systematic, tradeable premium between the implied event variance priced into short-dated options and the realized variance of the earnings move?

### Methodology

**`iv_stat_analysis.py`** processes the full option chain for each of the ~350 tickers across every earnings date in the sample. For each event, it tracks IV along three term-structure maturities (short / medium / long-dated) and three moneyness levels (ATM, upside wing, downside wing) from t-10 to t+10 trading days.

**Event Variance Decomposition** follows Bali & Hovakimian (2009) and Dubinsky et al. (2019):

```
sigma_short^2 * T_short  =  sigma_fwd^2 * (T_short - 1/252)  +  sigma_event^2 * (1/252)
```

The forward variance (short-to-long term structure) serves as the non-event proxy. The **Event VRP** is:

```
Event VRP  =  Var_event_implied  -  Var_event_realized
```

### Key Findings

Across 8,600+ earnings events:

- **The market systematically overprices earnings moves.** Implied event variance exceeds realized variance by ~2-4 vol points. Median overpricing ratio: ~60% of implied event variance.
- **IV builds up 5-7 days pre-earnings**, peaking at the last close before the announcement. Front-loaded crush: short-dated IV drops 70-80% of its abnormal component overnight.
- **Downside skew dominates**: the put wing carries more abnormal IV than the call wing, with the spread widening into earnings.
- **Persistence across events**: some names (NFLX, TSLA, META) show positive lag-1 autocorrelation in Event VRP -- overpricing at one earnings predicts overpricing at the next.
- **Term slope predicts crush magnitude**: steeper term structure forecasts larger post-earnings IV drops (t-stat > 5).

### Signal Construction

| Approach | Method | Output |
|----------|--------|--------|
| Rule-based (`signal_generator.py`) | Z-scored composite of abnormal IV, term slope, earnings skew | Trading signals |
| ML (`signal_generator_ml.py`) | Walk-forward GBM on 11 pre-event features, SHAP attribution | Probabilistic signals |

All features are strictly pre-event. ML uses expanding-window validation: for year Y, train on years < Y only.

---

## Usage

The project supports four workflows:

### 1. Earnings IV Study -- cross-sectional research across ~350 names

```bash
python iv_stat_analysis.py                  # Event VRP decomposition across ~350 tickers
python signal_generator.py                  # Rule-based trading signals
python signal_generator_ml.py               # Walk-forward GBM signals
```

### 2. Flagship Strategy Suite -- ETFs (Alpha Vantage data)

```bash
python flagship_final.py                    # Run all 7 strategies on QQQ
python flagship_final.py --symbol SPY       # Run on SPY
python flagship_final.py --carry-only       # Carry bucket only
python flagship_metrics.py --version 7      # Composite portfolio analytics
```

### 3. Flagship Strategy Suite -- Indices (OptionMetrics/CBOE data)

```bash
python flagship_final_cboe.py --symbol NDX  # Run all on NDX
python flagship_final_cboe.py --symbol SPX  # Run all on SPX
python flagship_metrics.py --version 7 --cboe  # Composite portfolio analytics
```

### 4. Single-Strategy Backtester -- any underlying

The `backtester/` engine and `run_backtest.py` runner work on any underlying with available option chain data. Strategies from the `strategies/` registry (earnings, rolling put-write, variance swap, calendar spread, dispersion) can be run independently on any ticker:

```bash
python run_backtest.py --symbol SPY --mode rolling    # Rolling put-write on SPY
python run_backtest.py --symbol AAPL --mode earnings  # Earnings straddle on AAPL
python run_backtest.py --symbol TSLA --mode varswap   # Variance swap on TSLA
```

## Requirements

```
pip install -r requirements.txt
```

```
numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, pyarrow, xlsxwriter, openpyxl, tqdm
```
