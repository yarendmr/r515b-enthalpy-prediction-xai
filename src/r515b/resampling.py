"""Training-only SMOGN generation with physical-range filtering."""

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from .config import SEED

def smogn_veri_uret(df_train_birlesik, hedef_kol, k, rel_thres, pert, under_samp):
    import smogn
    try:
        df_log = df_train_birlesik.copy()
        if (df_log["P"] <= 0).any() or (df_log["v"] <= 0).any():
            return None

        df_log["P"] = np.log(df_log["P"])
        df_log["v"] = np.log(df_log["v"])

        olcek_kolonlari = ["T", "P", "v", "s"]
        scaler = StandardScaler()
        df_log[olcek_kolonlari] = scaler.fit_transform(df_log[olcek_kolonlari])

        h_min = df_train_birlesik[hedef_kol].min()
        h_max = df_train_birlesik[hedef_kol].max()

        np.random.seed(SEED)
        df_res = smogn.smoter(
            data=df_log,
            y=hedef_kol,
            k=int(k),
            rel_thres=float(rel_thres),
            pert=float(pert),
            samp_method="balance",
            under_samp=bool(under_samp),
            replace=True
        )

        if df_res is None or len(df_res) == 0:
            return None

        df_res[olcek_kolonlari] = scaler.inverse_transform(df_res[olcek_kolonlari])
        df_res["P"] = np.exp(df_res["P"])
        df_res["v"] = np.exp(df_res["v"])
        df_res = df_res.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        T_lo, T_hi = df_train_birlesik["T"].min() - 5, df_train_birlesik["T"].max() + 5
        P_lo, P_hi = df_train_birlesik["P"].min() * 0.5, df_train_birlesik["P"].max() * 1.5
        v_lo, v_hi = df_train_birlesik["v"].min() * 0.5, df_train_birlesik["v"].max() * 1.5

        df_res = df_res[
            (df_res["T"] >= T_lo) & (df_res["T"] <= T_hi) &
            (df_res["P"] >= P_lo) & (df_res["P"] <= P_hi) &
            (df_res["v"] >= v_lo) & (df_res["v"] <= v_hi) &
            (df_res[hedef_kol] >= h_min) & (df_res[hedef_kol] <= h_max)
        ].reset_index(drop=True)

        if len(df_res) == 0:
            return None

        df_birlesik = pd.concat([df_train_birlesik, df_res], axis=0, ignore_index=True)
        df_birlesik = df_birlesik.drop_duplicates().reset_index(drop=True)

        if len(df_birlesik) < len(df_train_birlesik):
            return None

        return df_birlesik
    except Exception:
        return None
