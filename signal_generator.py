# signal_generator.py (compact style)
import os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr

BASE_DIR = r"C:\Users\antoi\Documents\Antoine\Projets_Python\Trading Vol on Earnings"
IN_EXCEL = os.path.join(BASE_DIR, "outputs", "earnings_iv_analysis.xlsx")
OUT_SIGNALS = os.path.join(BASE_DIR, "outputs", "signals_all.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "signals_by_symbol")
OUT_XLS = os.path.join(BASE_DIR, "outputs", "signals_analysis.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

def zsc(s): m,sig=s.mean(),s.std(); return (s-m)/sig if sig and sig>0 else pd.Series(0.0,index=s.index)

def ols(df,y,x,n):
    df=df.dropna(subset=[y]+x);
    if df.empty: return None,None
    m=sm.OLS(df[y],sm.add_constant(df[x])).fit(cov_type="HC3")
    coef=pd.DataFrame({"model":n,"var":m.params.index,"coef":m.params.values,"t":m.tvalues.values,"p":m.pvalues.values})
    summ=pd.DataFrame({"model":[n],"nobs":[m.nobs],"rsq":[m.rsquared],"rsq_adj":[m.rsquared_adj],"f":[m.fvalue],"f_p":[m.f_pvalue]})
    return coef,summ

def buckets(df,sc,targets,n=10):
    df=df.dropna(subset=[sc]+targets)
    if df.empty: return None
    df["b"]=pd.qcut(df[sc],n,labels=False,duplicates="drop")+1
    return df.groupby("b")[targets].mean().assign(count=df.groupby("b").size()).reset_index().assign(signal=sc)

def build():
    ef=pd.read_excel(IN_EXCEL,"EventFeatures_ATM")
    efa=pd.read_excel(IN_EXCEL,"EventFeatures_Tracks")
    for c in ["EventDate","AnchorDate","AnchorDate_plus1"]:
        if c in ef: ef[c]=pd.to_datetime(ef[c]).dt.normalize()
    efa["EventDate"]=pd.to_datetime(efa["EventDate"]).dt.normalize()
    piv=efa[efa["CaseID"]==1].pivot_table(index=["Symbol","EventDate"],columns="MnyTrackID",values="IV_pre")
    for k in [0,1,2]:
        if k not in piv: piv[k]=np.nan
    piv=piv[[0,1,2]].rename(columns={0:"IV_pre_m0",1:"IV_pre_m1",2:"IV_pre_m2"}).reset_index()
    df=ef.merge(piv,on=["Symbol","EventDate"],how="left")
    df["EM_skew_pre"]=0.5*(df["IV_pre_m1"]+df["IV_pre_m2"])-df["IV_pre_m0"]
    df["EM_skew_rel"]=df["EM_skew_pre"]/df["IV_pre_m0"]
    df["RealizedMove_1d"] = df["R_0_1"].abs()
    df["ImpliedMove_1d"] = df["EV_1d"]
    df["IVCrush_Impact"] = -df["Crush_1d"] * np.sqrt(1 / 252)
    df["PnL_proxy"] = df["IVCrush_Impact"] + df["ImpliedMove_1d"] - df["RealizedMove_1d"]
    # Event_VRP is the primary target (computed in iv_stat_analysis.py, carried in EventFeatures_ATM)
    if "Event_VRP" not in df.columns:
        # Fallback: compute if not present in the input sheet
        if "Var_event_implied" in df.columns and "Var_event_realized" in df.columns:
            df["Event_VRP"] = df["Var_event_implied"] - df["Var_event_realized"]
        elif "IV_pre_long" in df.columns and "TTM_pre_yrs" in df.columns:
            var_total = df["IV_pre"] ** 2 * df["TTM_pre_yrs"]
            var_regular = df["IV_pre_long"] ** 2
            df["Var_event_implied"] = (var_total - var_regular * (df["TTM_pre_yrs"] - 1.0/252.0)) * 252.0
            df["Var_event_realized"] = df["R_0_1"] ** 2 * 252.0
            df["Event_VRP"] = df["Var_event_implied"] - df["Var_event_realized"]
    for c in ["IV_abn_pre","TermSlope_pre","IV_pre","EM_skew_pre","EM_skew_rel"]:
        df[c+"_z"]=zsc(df[c])
    df["ShortScore_z"]=df["TermSlope_pre_z"].copy()  # Pure TermSlope signal for filtering
    df["LongScore_z"]=(-df["TermSlope_pre_z"]).copy()  # Inverse for long vol
    df["ShortScore_q"]=pd.qcut(df["ShortScore_z"].rank(method="first"),5,labels=False)+1
    df["LongScore_q"]=pd.qcut(df["LongScore_z"].rank(method="first"),5,labels=False)+1
    short_map={1:0.0,2:0.5,3:1.0,4:1.5,5:2.0}; long_map={1:2.0,2:1.5,3:1.0,4:0.5,5:0.0}
    df["VegaMult_short"]=df["ShortScore_q"].map(short_map).astype(float)
    df["VegaMult_long"]=df["LongScore_q"].map(long_map).astype(float)
    cols=[c for c in df.columns if c in [
        "Symbol","EventDate","AnchorDate","Maturity_pre","TTM_pre_yrs","IV_pre","IV_abn_pre","TermSlope_pre","EV_1d",
        "EM_skew_pre","EM_skew_rel","IV_pre_z","IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z","EM_skew_rel_z",
        "ShortScore_z","ShortScore_q","VegaMult_short","LongScore_z","LongScore_q","VegaMult_long",
        "Crush_1d","IV_abn_post","RealizedMove_1d","ImpliedMove_1d","IVCrush_Impact","PnL_proxy",
        "Event_VRP","Var_event_implied","Var_event_realized",
    ]]
    sig=df[cols].sort_values(["Symbol","EventDate"]).reset_index(drop=True)
    sig.to_csv(OUT_SIGNALS,index=False)
    for s,g in sig.groupby("Symbol"): g.to_csv(os.path.join(OUT_DIR,f"{s}_signals.csv"),index=False)
    req_cols=["Crush_1d","PnL_proxy","IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"]
    if "Event_VRP" in df.columns: req_cols.append("Event_VRP")
    reg_df=df.dropna(subset=req_cols).copy()
    models=[("Crush_IVabn","Crush_1d",["IV_abn_pre_z"]),("Crush_Term","Crush_1d",["TermSlope_pre_z"]),
            ("Crush_EM","Crush_1d",["EM_skew_pre_z"]),("Crush_All","Crush_1d",["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"]),
            ("PnL_IVabn","PnL_proxy",["IV_abn_pre_z"]),("PnL_Term","PnL_proxy",["TermSlope_pre_z"]),
            ("PnL_EM","PnL_proxy",["EM_skew_pre_z"]),("PnL_All","PnL_proxy",["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"])]
    # Add Event_VRP regressions if available
    if "Event_VRP" in reg_df.columns:
        models+=[("EVRP_IVabn","Event_VRP",["IV_abn_pre_z"]),("EVRP_Term","Event_VRP",["TermSlope_pre_z"]),
                 ("EVRP_EM","Event_VRP",["EM_skew_pre_z"]),("EVRP_All","Event_VRP",["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z"])]
    coefs=[]; sums=[]
    for n,y,x in models:
        c,s=ols(reg_df,y,x,n)
        if c is not None: coefs.append(c); sums.append(s)
    bucket_cols=["Crush_1d","PnL_proxy","EV_1d","RealizedMove_1d"]
    if "Event_VRP" in df.columns: bucket_cols.append("Event_VRP")
    b1=buckets(df,"IV_abn_pre_z",bucket_cols); b2=buckets(df,"TermSlope_pre_z",bucket_cols)
    b3=buckets(df,"EM_skew_pre_z",bucket_cols); b4=buckets(df,"ShortScore_z",bucket_cols)
    b5=buckets(df,"LongScore_z",bucket_cols)
    corr_cols_list=["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z","PnL_proxy","Crush_1d"]
    if "Event_VRP" in df.columns: corr_cols_list.append("Event_VRP")
    corr=df[[c for c in corr_cols_list if c in df.columns]].corr()

    # --- IC Analysis (Information Coefficient) ---
    signal_cols=["IV_abn_pre_z","TermSlope_pre_z","EM_skew_pre_z","ShortScore_z","LongScore_z"]
    target_cols=["Crush_1d","PnL_proxy"]
    if "Event_VRP" in df.columns: target_cols.append("Event_VRP")
    ic_rows=[]
    for sc in signal_cols:
        for tgt in target_cols:
            sub=df.dropna(subset=[sc,tgt])
            if len(sub)<30: continue
            ic,pval=spearmanr(sub[sc],sub[tgt])
            n=len(sub)
            tstat=ic*np.sqrt((n-2)/(1-ic**2)) if abs(ic)<1 else np.inf
            ic_rows.append({"Signal":sc,"Target":tgt,"IC":round(ic,4),"IC_tstat":round(tstat,2),"IC_pvalue":round(pval,4),"N_obs":n})
    ic_table=pd.DataFrame(ic_rows)

    # --- Quintile Monotonicity Check ---
    q_dfs=[]
    for sc in signal_cols:
        sub=df.dropna(subset=[sc]).copy()
        if len(sub)<50: continue
        sub["Quintile"]=pd.qcut(sub[sc].rank(method="first"),5,labels=False,duplicates="drop")+1
        q=sub.groupby("Quintile")[target_cols].mean()
        q["Count"]=sub.groupby("Quintile").size()
        q["Signal"]=sc
        q_dfs.append(q.reset_index())
    quintile_mono=pd.concat(q_dfs,ignore_index=True) if q_dfs else pd.DataFrame()

    with pd.ExcelWriter(OUT_XLS) as w:
        sig.to_excel(w,"signals",index=False)
        (pd.concat(coefs) if coefs else pd.DataFrame()).to_excel(w,"reg_coef",index=False)
        (pd.concat(sums) if sums else pd.DataFrame()).to_excel(w,"reg_summary",index=False)
        for name,b in [("b_IVabn",b1),("b_Term",b2),("b_EMskew",b3),("b_Short",b4),("b_Long",b5)]:
            (b if b is not None else pd.DataFrame()).to_excel(w,name,index=False)
        corr.to_excel(w,"corr")
        if not ic_table.empty: ic_table.to_excel(w,"IC_Analysis",index=False)
        if not quintile_mono.empty: quintile_mono.to_excel(w,"Quintile_Monotonicity",index=False)
    print("[OK] signals, IC analysis, and quintile monotonicity built")

if __name__=="__main__": build()
