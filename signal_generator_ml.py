# ============================================================
# signal_generator_ml.py  (FIXED)
# ============================================================
# Build ML-based earnings IV-mispricing signals using:
#   - EventFeatures_ATM
#   - SkewBySide_Events (upside & downside skew) if available
#
# NO FORWARD-LOOKING FEATURES IN X:
#   X = only pre-earnings information (IV_abn_pre, TermSlope_pre, EV_1d,
#       TTM_pre_yrs, upside/downside skew levels & abnormal skew if present).
#   y = Event_VRP (implied event variance - realized event variance) if available,
#       else PnL_proxy (legacy). Event_VRP is a cleaner, more economically meaningful
#       target that directly measures the tradeable earnings event premium.
#
# ML:
#   - GradientBoostingRegressor
#   - Rolling by AnchorYear: for year Y, train on previous N years only
#
# OUTPUT:
#   outputs/signals_all_ml.csv
#   outputs/signals_by_symbol_ml/*
#   outputs/signals_ml_analysis.xlsx (diagnostics)
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr


# ------------------------------------------------------------------
# CONFIG: set base directory
# ------------------------------------------------------------------
# Option 1: dynamic (script directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If you prefer your hard-coded path, uncomment this and adjust:
# BASE_DIR = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings"

IN_EXCEL    = os.path.join(BASE_DIR, "outputs", "earnings_iv_analysis.xlsx")
OUT_SIGNALS = os.path.join(BASE_DIR, "outputs", "signals_all_ml.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs", "signals_by_symbol_ml")
OUT_XLS     = os.path.join(BASE_DIR, "outputs", "signals_ml_analysis.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)


def buckets(df: pd.DataFrame, col: str, targets, n: int = 10):
    """
    Simple decile buckets for diagnostics.

    df: DataFrame
    col: score column name
    targets: list of numeric columns to average in each bucket
    """
    if col not in df.columns:
        return None
    df = df.dropna(subset=[col] + targets)
    if df.empty:
        return None
    df = df.copy()
    df["b"] = pd.qcut(
        df[col], n, labels=False, duplicates="drop"
    ) + 1
    out = (
        df.groupby("b")[targets]
        .mean()
        .assign(count=df.groupby("b").size())
        .reset_index()
    )
    out["signal"] = col
    return out


def compute_ic_table(df, signal_cols, target_cols):
    """Compute Information Coefficient (Spearman rank correlation) table."""
    rows = []
    for sig in signal_cols:
        for tgt in target_cols:
            sub = df.dropna(subset=[sig, tgt])
            if len(sub) < 30:
                continue
            ic, pval = spearmanr(sub[sig], sub[tgt])
            n = len(sub)
            tstat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else np.inf
            rows.append({
                "Signal": sig, "Target": tgt,
                "IC": round(ic, 4), "IC_tstat": round(tstat, 2),
                "IC_pvalue": round(pval, 4), "N_obs": n,
            })
    return pd.DataFrame(rows)


def build():
    # --------------------------------------------------------------
    # 1) Load EventFeatures_ATM & SkewBySide_Events
    # --------------------------------------------------------------
    ef = pd.read_excel(IN_EXCEL, sheet_name="EventFeatures_ATM")

    try:
        skew = pd.read_excel(IN_EXCEL, sheet_name="SkewBySide_Events")
    except ValueError:
        skew = None

    # Normalise dates
    for c in ["EventDate", "AnchorDate", "AnchorDate_plus1"]:
        if c in ef.columns:
            ef[c] = pd.to_datetime(ef[c]).dt.normalize()

    if skew is not None:
        skew["EventDate"] = pd.to_datetime(skew["EventDate"]).dt.normalize()
        if "AnchorDate" in skew.columns:
            skew["AnchorDate"] = pd.to_datetime(skew["AnchorDate"]).dt.normalize()

    # Focus on CaseID = 1 (short-dated case) if present
    if "CaseID" in ef.columns:
        ef = ef[ef["CaseID"] == 1].copy()
    if skew is not None and "CaseID" in skew.columns:
        skew = skew[skew["CaseID"] == 1].copy()

    # --------------------------------------------------------------
    # 2) Merge skew data (if available) into ef
    # --------------------------------------------------------------
    if skew is not None:
        skew_cols = [
            "Symbol",
            "EventDate",
            "Skew_IV_pre_UP",
            "Skew_IV_pre_DOWN",
            "Skew_IV_abn_pre_UP",
            "Skew_IV_abn_pre_DOWN",
        ]
        skew_cols = [c for c in skew_cols if c in skew.columns]

        if skew_cols:
            ef = ef.merge(
                skew[skew_cols],
                on=["Symbol", "EventDate"],
                how="left",
            )

    # Derived skew composites (if underlying columns exist):
    if (
        "Skew_IV_pre_UP" in ef.columns
        and "Skew_IV_pre_DOWN" in ef.columns
        and "IV_pre" in ef.columns
    ):
        # Here Skew_IV_pre_* is (wing IV - ATM IV). Average gives EM-style skew.
        ef["EM_skew_pre"] = 0.5 * (ef["Skew_IV_pre_UP"] + ef["Skew_IV_pre_DOWN"])
        ef["EM_skew_rel"] = ef["EM_skew_pre"] / ef["IV_pre"]

    if "Skew_IV_abn_pre_UP" in ef.columns and "Skew_IV_abn_pre_DOWN" in ef.columns:
        ef["Skew_DownMinusUp_abn"] = (
            ef["Skew_IV_abn_pre_DOWN"] - ef["Skew_IV_abn_pre_UP"]
        )

    # --------------------------------------------------------------
    # 3) Build targets – post-event info ONLY as y
    # --------------------------------------------------------------
    for c in ["R_0_1", "EV_1d", "Crush_1d"]:
        if c not in ef.columns:
            raise ValueError(f"Required column '{c}' not found in EventFeatures_ATM")

    ef["RealizedMove_1d"] = ef["R_0_1"].abs()
    ef["ImpliedMove_1d"] = ef["EV_1d"]
    ef["IVCrush_Impact"] = -ef["Crush_1d"] * np.sqrt(1.0 / 252.0)

    ef["PnL_proxy"] = (
        ef["IVCrush_Impact"] + ef["RealizedMove_1d"] - ef["ImpliedMove_1d"]
    )

    # Event_VRP is the primary target (from iv_stat_analysis.py EventFeatures_ATM)
    # Fallback: compute if not present in the input sheet
    if "Event_VRP" not in ef.columns:
        if "IV_pre_long" in ef.columns and "TTM_pre_yrs" in ef.columns:
            var_total = ef["IV_pre"] ** 2 * ef["TTM_pre_yrs"]
            var_regular = ef["IV_pre_long"] ** 2
            ef["Var_event_implied"] = (var_total - var_regular * (ef["TTM_pre_yrs"] - 1.0/252.0)) * 252.0
            ef["Var_event_realized"] = ef["R_0_1"] ** 2 * 252.0
            ef["Event_VRP"] = ef["Var_event_implied"] - ef["Var_event_realized"]

    # Use Event_VRP as primary target if available, fallback to PnL_proxy
    TARGET_COL = "Event_VRP" if "Event_VRP" in ef.columns and ef["Event_VRP"].notna().sum() > 100 else "PnL_proxy"
    print(f"[INFO] ML target column: {TARGET_COL}")

    ef = ef.dropna(subset=[TARGET_COL]).copy()

    # --------------------------------------------------------------
    # 4) Build pre-event feature matrix X (no forward-looking)
    # --------------------------------------------------------------
    # Allowed features = purely pre-event quantities:
    candidate_feats = [
        "IV_abn_pre",
        "TermSlope_pre",
        "EV_1d",
        "TTM_pre_yrs",
        "EM_skew_pre",
        "EM_skew_rel",
        "Skew_IV_pre_UP",
        "Skew_IV_pre_DOWN",
        "Skew_IV_abn_pre_UP",
        "Skew_IV_abn_pre_DOWN",
        "Skew_DownMinusUp_abn",
    ]
    FEAT_COLS = [c for c in candidate_feats if c in ef.columns]

    if not FEAT_COLS:
        raise ValueError(
            "No usable feature columns found. Check iv_stat_analysis outputs "
            "and skew merge."
        )

    if "AnchorDate" not in ef.columns:
        raise ValueError("AnchorDate not found in EventFeatures_ATM")
    ef["AnchorYear"] = pd.to_datetime(ef["AnchorDate"]).dt.year

    # --------------------------------------------------------------
    # 5) Rolling ML by year: train on past N years only
    # --------------------------------------------------------------
    ef["ShortScore_ml"] = np.nan
    ef["LongScore_ml"] = np.nan

    min_years_history = 2
    min_obs = 200  # minimum events to train a model

    years = sorted(ef["AnchorYear"].dropna().unique())

    shap_rows = []
    year_diag = []

    for y in years:
        anchor_start = pd.Timestamp(year=int(y), month=1, day=1)
        train_start = anchor_start - pd.DateOffset(years=min_years_history)

        train_mask = (ef["AnchorDate"] >= train_start) & (
            ef["AnchorDate"] < anchor_start
        )
        test_mask = ef["AnchorYear"] == y

        df_tr = ef.loc[train_mask, FEAT_COLS + [TARGET_COL]].dropna(
            subset=[TARGET_COL]
        )
        df_te = ef.loc[test_mask, FEAT_COLS]

        if df_tr.shape[0] < min_obs:
            # Not enough history for a robust model for year y
            continue

        # Median imputation on training features
        med = df_tr[FEAT_COLS].median()
        X_train = df_tr[FEAT_COLS].fillna(med)
        y_train = df_tr[TARGET_COL].values

        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        X_test = df_te[FEAT_COLS].fillna(med)
        preds = model.predict(X_test)

        ef.loc[test_mask, "ShortScore_ml"] = preds
        ef.loc[test_mask, "LongScore_ml"] = -preds

        # Collect SHAP values and feature importances
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            shap_importance = pd.DataFrame({
                "feature": FEAT_COLS,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
                "year": y,
            })
            shap_rows.append(shap_importance)
        except ImportError:
            pass

        # Per-year OOS diagnostics
        oos_ic, oos_pval = spearmanr(ef.loc[test_mask, TARGET_COL].values, preds)
        year_diag.append({
            "year": int(y),
            "n_train": len(df_tr),
            "n_test": int(test_mask.sum()),
            "train_r2": model.score(X_train, y_train),
            "oos_ic": oos_ic,
            "oos_ic_pval": oos_pval,
            "features_used": ", ".join(FEAT_COLS),
        })

    # --------------------------------------------------------------
    # 5b) Post-loop: feature importance & IC analysis
    # --------------------------------------------------------------

    # Sklearn feature importance (from last model as representative)
    fi = None
    last_model = None
    if year_diag:  # at least one year was trained
        # Re-train on the last year's data to get feature importances
        # (model variable may not persist from loop scope in all cases)
        last_y = year_diag[-1]["year"]
        last_anchor = pd.Timestamp(year=last_y, month=1, day=1)
        last_train_start = last_anchor - pd.DateOffset(years=min_years_history)
        last_train_mask = (ef["AnchorDate"] >= last_train_start) & (ef["AnchorDate"] < last_anchor)
        last_tr = ef.loc[last_train_mask, FEAT_COLS + [TARGET_COL]].dropna(subset=[TARGET_COL])
        if len(last_tr) >= min_obs:
            last_med = last_tr[FEAT_COLS].median()
            last_model = GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )
            last_model.fit(last_tr[FEAT_COLS].fillna(last_med), last_tr[TARGET_COL].values)
            fi = pd.DataFrame({
                "Feature": FEAT_COLS,
                "Importance": last_model.feature_importances_,
            }).sort_values("Importance", ascending=False)

    # IC analysis
    ml_signal_cols = ["ShortScore_z", "LongScore_z"]
    # Note: scores are computed in section 6 below, so we defer IC to after that

    # --------------------------------------------------------------
    # 6) Map ML scores to ShortScore_z / LongScore_z and vega multipliers
    # --------------------------------------------------------------
    # We use raw ML predictions as "z-scores" for the backtester;
    # the rules engine will build buckets and weights from them.

    ef["ShortScore_z"] = ef["ShortScore_ml"]
    ef["LongScore_z"] = ef["LongScore_ml"]

    # Quantile buckets
    mask_short = ef["ShortScore_z"].notna()
    if mask_short.any():
        ef.loc[mask_short, "ShortScore_q"] = (
            pd.qcut(
                ef.loc[mask_short, "ShortScore_z"].rank(method="first"),
                5,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

    mask_long = ef["LongScore_z"].notna()
    if mask_long.any():
        ef.loc[mask_long, "LongScore_q"] = (
            pd.qcut(
                ef.loc[mask_long, "LongScore_z"].rank(method="first"),
                5,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

    short_map = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0}
    long_map = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.5, 5: 0.0}

    if "ShortScore_q" not in ef.columns:
        ef["ShortScore_q"] = np.nan
    if "LongScore_q" not in ef.columns:
        ef["LongScore_q"] = np.nan

    ef["VegaMult_short"] = ef["ShortScore_q"].map(short_map).astype(float)
    ef["VegaMult_long"] = ef["LongScore_q"].map(long_map).astype(float)

    # IC analysis (after ShortScore_z / LongScore_z are populated)
    ml_signal_cols = ["ShortScore_z", "LongScore_z"]
    ml_target_cols = ["Crush_1d", "PnL_proxy"]
    if "Event_VRP" in ef.columns:
        ml_target_cols.append("Event_VRP")
    ic_table = compute_ic_table(ef, ml_signal_cols, ml_target_cols)

    # --------------------------------------------------------------
    # 7) Final signals CSV – interface for backtester
    # --------------------------------------------------------------
    cols_to_keep = [
        # keys / timing
        "Symbol",
        "EventDate",
        "AnchorDate",
        "Maturity_pre",
        "TTM_pre_yrs",
        # pre features
        "IV_pre",
        "IV_abn_pre",
        "TermSlope_pre",
        "EV_1d",
        "EM_skew_pre",
        "EM_skew_rel",
        "Skew_IV_pre_UP",
        "Skew_IV_pre_DOWN",
        "Skew_IV_abn_pre_UP",
        "Skew_IV_abn_pre_DOWN",
        "Skew_DownMinusUp_abn",
        # ML scores & vega multipliers
        "ShortScore_z",
        "ShortScore_q",
        "VegaMult_short",
        "LongScore_z",
        "LongScore_q",
        "VegaMult_long",
        # target / diagnostics
        "Crush_1d",
        "IV_abn_post",
        "RealizedMove_1d",
        "ImpliedMove_1d",
        "IVCrush_Impact",
        "PnL_proxy",
        "Event_VRP",
        "Var_event_implied",
        "Var_event_realized",
    ]

    cols = [c for c in cols_to_keep if c in ef.columns]
    sig = ef[cols].sort_values(["Symbol", "EventDate"]).reset_index(drop=True)

    sig.to_csv(OUT_SIGNALS, index=False)

    # Per-symbol files (for debugging)
    for sym, g in sig.groupby("Symbol"):
        g.to_csv(os.path.join(OUT_DIR, f"{sym}_signals_ml.csv"), index=False)

    # --------------------------------------------------------------
    # 8) Diagnostics workbook: buckets & correlations
    # --------------------------------------------------------------
    bucket_targets = ["Crush_1d", "PnL_proxy", "EV_1d", "RealizedMove_1d"]
    if "Event_VRP" in ef.columns:
        bucket_targets.append("Event_VRP")

    b_iv    = buckets(ef, "IV_abn_pre", bucket_targets)
    b_term  = buckets(ef, "TermSlope_pre", bucket_targets)
    b_em    = buckets(ef, "EM_skew_pre", bucket_targets) if "EM_skew_pre" in ef.columns else None
    b_short = buckets(ef, "ShortScore_z", bucket_targets)
    b_long  = buckets(ef, "LongScore_z", bucket_targets)

    corr_cols = [
        c
        for c in [
            "IV_abn_pre",
            "TermSlope_pre",
            "EM_skew_pre",
            "Skew_IV_abn_pre_UP",
            "Skew_IV_abn_pre_DOWN",
            "Skew_DownMinusUp_abn",
            "PnL_proxy",
            "Event_VRP",
            "Crush_1d",
            "ShortScore_z",
            "LongScore_z",
        ]
        if c in ef.columns
    ]
    corr = ef[corr_cols].corr() if corr_cols else pd.DataFrame()

    with pd.ExcelWriter(OUT_XLS) as w:
        sig.to_excel(w, "signals_ml", index=False)
        if b_iv is not None:
            b_iv.to_excel(w, "bucket_IVabn", index=False)
        if b_term is not None:
            b_term.to_excel(w, "bucket_Term", index=False)
        if b_em is not None:
            b_em.to_excel(w, "bucket_EMskew", index=False)
        if b_short is not None:
            b_short.to_excel(w, "bucket_ShortML", index=False)
        if b_long is not None:
            b_long.to_excel(w, "bucket_LongML", index=False)
        if not corr.empty:
            corr.to_excel(w, "corr", index=True)

        # SHAP importance
        if shap_rows:
            df_shap = pd.concat(shap_rows, ignore_index=True)
            df_shap.to_excel(w, "SHAP_Importance", index=False)
            # Aggregate across years
            shap_agg = df_shap.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).reset_index()
            shap_agg.columns = ["Feature", "Mean_Abs_SHAP"]
            shap_agg.to_excel(w, "SHAP_Summary", index=False)

        # Year diagnostics
        if year_diag:
            pd.DataFrame(year_diag).to_excel(w, "Model_Diagnostics", index=False)

        # IC analysis
        if not ic_table.empty:
            ic_table.to_excel(w, "IC_Analysis", index=False)

        # Sklearn feature importance (from last trained model)
        if fi is not None:
            fi.to_excel(w, "Feature_Importance", index=False)

    print("[OK] ML signals and analysis built:")
    print(f"  -> {OUT_SIGNALS}")
    print(f"  -> {OUT_XLS}")


if __name__ == "__main__":
    build()
