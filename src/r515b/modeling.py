"""Tree regressors, model selection, scoring, and held-out evaluation."""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import HEDEF, MODEL_GRIDLERI, OZELLIKLER, SEED, SMOGN_GRID
from .data import scaler_fit_transform
from .resampling import smogn_veri_uret

def guvenli_int_or_none(x):
    if x is None or pd.isna(x):
        return None
    return int(x)

def metrik_hesapla(y_gercek, y_tahmin):
    y_g = np.array(y_gercek)
    y_t = np.array(y_tahmin)
    residual = y_g - y_t
    lo = np.quantile(y_g, 0.15)
    hi = np.quantile(y_g, 0.85)
    mask_tail = (y_g <= lo) | (y_g >= hi)
    return {
        "R2": float(r2_score(y_g, y_t)),
        "RMSE": float(np.sqrt(mean_squared_error(y_g, y_t))),
        "MAE": float(mean_absolute_error(y_g, y_t)),
        "Tail_MAE": float(mean_absolute_error(y_g[mask_tail], y_t[mask_tail])),
        "Res_Ort": float(residual.mean()),
        "Res_Std": float(residual.std()),
        "Maks_Hata": float(np.abs(residual).max()),
        "Pm1_kJ_pct": float((np.abs(residual) <= 1.0).mean() * 100.0),
        "Pm2_kJ_pct": float((np.abs(residual) <= 2.0).mean() * 100.0)
    }

def genel_skor(m):
    return float(m["R2"] - 0.10 * m["RMSE"] - 0.05 * m["MAE"])

def model_kur(model_adi, prm):
    if model_adi == "GB":
        return GradientBoostingRegressor(
            n_estimators=int(prm["n_estimators"]),
            learning_rate=float(prm["learning_rate"]),
            max_depth=int(prm["max_depth"]),
            subsample=float(prm["subsample"]),
            random_state=SEED
        )
    if model_adi == "RF":
        return RandomForestRegressor(
            n_estimators=int(prm["n_estimators"]),
            max_depth=guvenli_int_or_none(prm.get("max_depth", None)),
            min_samples_leaf=int(prm["min_samples_leaf"]),
            max_features=prm["max_features"],
            random_state=SEED,
            n_jobs=-1
        )
    if model_adi == "ET":
        return ExtraTreesRegressor(
            n_estimators=int(prm["n_estimators"]),
            max_depth=guvenli_int_or_none(prm.get("max_depth", None)),
            min_samples_leaf=int(prm["min_samples_leaf"]),
            max_features=prm["max_features"],
            random_state=SEED,
            n_jobs=-1
        )
    raise ValueError("Bilinmeyen model")

def cv_metrikleri(model_adi, prm, X_df, y_sr, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    skorlar = []
    for tr_idx, vl_idx in kf.split(X_df):
        X_tr = X_df.iloc[tr_idx]
        X_vl = X_df.iloc[vl_idx]
        y_tr = y_sr.iloc[tr_idx]
        y_vl = y_sr.iloc[vl_idx]
        scaler, X_tr_s, X_vl_s = scaler_fit_transform(X_tr, X_vl)
        model = model_kur(model_adi, prm)
        model.fit(X_tr_s, y_tr)
        y_hat = model.predict(X_vl_s)
        skorlar.append(metrik_hesapla(y_vl, y_hat))
    df_cv = pd.DataFrame(skorlar)
    return {
        "R2_cv_mean": float(df_cv["R2"].mean()),
        "RMSE_cv_mean": float(df_cv["RMSE"].mean()),
        "MAE_cv_mean": float(df_cv["MAE"].mean()),
        "Tail_MAE_cv_mean": float(df_cv["Tail_MAE"].mean())
    }

def orijinal_tarama(X_train, y_train, X_valid, y_valid):
    sonuclar = []
    for model_adi, grid in MODEL_GRIDLERI.items():
        print(f"\n{model_adi} — orijinal tarama")
        for prm in ParameterGrid(grid):
            cv = cv_metrikleri(model_adi, prm, X_train, y_train, n_splits=3)
            scaler, X_tr_s, X_vl_s = scaler_fit_transform(X_train, X_valid)
            model = model_kur(model_adi, prm)
            model.fit(X_tr_s, y_train)
            y_hat = model.predict(X_vl_s)
            m = metrik_hesapla(y_valid, y_hat)
            sonuclar.append({
                "Senaryo": "Orijinal",
                "Model": model_adi,
                **prm,
                **cv,
                "R2_valid": round(m["R2"], 5),
                "RMSE_valid": round(m["RMSE"], 5),
                "MAE_valid": round(m["MAE"], 5),
                "Tail_MAE_valid": round(m["Tail_MAE"], 5),
                "Genel_Skor": round(genel_skor(m), 5)
            })
    df = pd.DataFrame(sonuclar)
    print("\nORIJINAL TOP 5 TABLOLARI")
    for model_adi in ["GB", "RF", "ET"]:
        print(f"\nEn iyi {model_adi} — Orijinal")
        display(
            df[df["Model"] == model_adi]
            .sort_values(["R2_valid", "RMSE_valid"], ascending=[False, True])
            .head(5)
        )
    return df

def smogn_parametre_sec(X_train, y_train, X_valid, y_valid):
    df_train_birlesik = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)

    proxy_prm = {
        "n_estimators": 180,
        "learning_rate": 0.07,
        "max_depth": 3,
        "subsample": 0.90
    }

    kayitlar = []

    print("\nSMOGN parametre secimi basladi...")
    for idx, (k, thres, pert, under_samp) in enumerate(SMOGN_GRID, 1):
        df_aug = smogn_veri_uret(df_train_birlesik, HEDEF, k, thres, pert, under_samp)

        if df_aug is None:
            print(f"[{idx}/{len(SMOGN_GRID)}] k={k}, th={thres}, pert={pert}, under={under_samp} -> gecersiz")
            continue

        X_aug = df_aug[OZELLIKLER].copy()
        y_aug = df_aug[HEDEF].copy()

        scaler, X_aug_s, X_vl_s = scaler_fit_transform(X_aug, X_valid)
        model = model_kur("GB", proxy_prm)
        model.fit(X_aug_s, y_aug)
        y_hat = model.predict(X_vl_s)
        m = metrik_hesapla(y_valid, y_hat)

        kayitlar.append({
            "k": int(k),
            "rel_thres": float(thres),
            "pert": float(pert),
            "under_samp": bool(under_samp),
            "n_aug": int(len(df_aug)),
            "R2_valid": round(m["R2"], 5),
            "RMSE_valid": round(m["RMSE"], 5),
            "MAE_valid": round(m["MAE"], 5),
            "Tail_MAE_valid": round(m["Tail_MAE"], 5)
        })

        print(
            f"[{idx}/{len(SMOGN_GRID)}] "
            f"k={k}, th={thres}, pert={pert}, under={under_samp} -> "
            f"n={len(df_aug)}, R2={m['R2']:.4f}, Tail_MAE={m['Tail_MAE']:.4f}"
        )

    df_smogn_param = pd.DataFrame(kayitlar)

    df_filtreli = df_smogn_param[
        (df_smogn_param["R2_valid"] > 0.97) &
        (df_smogn_param["RMSE_valid"] < 2.5) &
        (df_smogn_param["MAE_valid"] < 1.8)
    ].copy()

    if len(df_filtreli) == 0:
        print("\n⚠️ Sıkı filtreyle uygun SMOGN adayı bulunamadi. Daha gevsek filtre uygulanacak.")
        df_filtreli = df_smogn_param[(df_smogn_param["R2_valid"] > 0.90)].copy()

    df_filtreli = df_filtreli.sort_values(
        ["Tail_MAE_valid", "MAE_valid", "RMSE_valid"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    print("\nEN IYI SMOGN PARAMETRELERI")
    display(df_filtreli.head(10))

    return df_filtreli, df_smogn_param

def smogn_model_tarama(X_train, y_train, X_valid, y_valid, df_smogn_param_secili):
    df_train_birlesik = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    en_iyi_smognler = df_smogn_param_secili.head(3).copy()
    sonuclar = []

    for _, sm_row in en_iyi_smognler.iterrows():
        k = int(sm_row["k"])
        th = float(sm_row["rel_thres"])
        pert = float(sm_row["pert"])
        under_samp = bool(sm_row["under_samp"])

        df_aug = smogn_veri_uret(df_train_birlesik, HEDEF, k, th, pert, under_samp)
        if df_aug is None:
            continue

        X_aug = df_aug[OZELLIKLER].copy()
        y_aug = df_aug[HEDEF].copy()

        print(f"\nSMOGN aday: k={k}, rel_thres={th}, pert={pert}, under_samp={under_samp}, n={len(df_aug)}")

        for model_adi, grid in MODEL_GRIDLERI.items():
            for prm in ParameterGrid(grid):
                scaler, X_aug_s, X_vl_s = scaler_fit_transform(X_aug, X_valid)
                model = model_kur(model_adi, prm)
                model.fit(X_aug_s, y_aug)
                y_hat = model.predict(X_vl_s)
                m = metrik_hesapla(y_valid, y_hat)

                sonuclar.append({
                    "Senaryo": "SMOGN",
                    "Model": model_adi,
                    "k": k,
                    "rel_thres": th,
                    "pert": pert,
                    "under_samp": under_samp,
                    "n_aug": int(len(df_aug)),
                    **prm,
                    "R2_valid": round(m["R2"], 5),
                    "RMSE_valid": round(m["RMSE"], 5),
                    "MAE_valid": round(m["MAE"], 5),
                    "Tail_MAE_valid": round(m["Tail_MAE"], 5),
                    "Genel_Skor": round(genel_skor(m), 5)
                })

    df = pd.DataFrame(sonuclar)

    df = df[(df["R2_valid"] > 0.90) & (df["RMSE_valid"] < 2.5)].copy()

    print("\nSMOGN TOP 5 TABLOLARI")
    for model_adi in ["GB", "RF", "ET"]:
        alt = df[df["Model"] == model_adi]
        print(f"\nEn iyi {model_adi} — SMOGN")
        if len(alt) > 0:
            display(
                alt.sort_values(
                    ["Tail_MAE_valid", "MAE_valid", "RMSE_valid"],
                    ascending=[True, True, True]
                ).head(5)
            )
        else:
            print("Sonuc yok.")

    return df

def en_iyi_orijinal_kayit(df, model_adi):
    return (
        df[df["Model"] == model_adi]
        .sort_values(["R2_valid", "RMSE_valid"], ascending=[False, True])
        .iloc[0]
    )

def en_iyi_smogn_kayitlari(df, model_adi):
    return (
        df[df["Model"] == model_adi]
        .sort_values(["Tail_MAE_valid", "MAE_valid", "RMSE_valid"], ascending=[True, True, True])
        .copy()
    )

def final_modeli_hazirla(kayit, senaryo, X_train_full, y_train_full, X_test, y_test):
    model_adi = kayit["Model"]

    if senaryo == "Orijinal":
        X_fit = X_train_full.copy()
        y_fit = y_train_full.copy()
    else:
        df_train_full_birlesik = pd.concat([X_train_full.reset_index(drop=True), y_train_full.reset_index(drop=True)], axis=1)
        df_aug = smogn_veri_uret(
            df_train_full_birlesik,
            HEDEF,
            int(kayit["k"]),
            float(kayit["rel_thres"]),
            float(kayit["pert"]),
            bool(kayit["under_samp"])
        )
        if df_aug is None or len(df_aug) == 0:
            return None
        X_fit = df_aug[OZELLIKLER].copy()
        y_fit = df_aug[HEDEF].copy()

    prm = {}
    for col in kayit.index:
        if col in ["n_estimators", "learning_rate", "max_depth", "subsample", "min_samples_leaf", "max_features"]:
            prm[col] = kayit[col]

    scaler = StandardScaler()
    scaler.fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_test_s = scaler.transform(X_test)

    model = model_kur(model_adi, prm)
    model.fit(X_fit_s, y_fit)
    y_hat = model.predict(X_test_s)
    m = metrik_hesapla(y_test, y_hat)

    return {
        "Model": model_adi,
        "Senaryo": senaryo,
        "kayit": kayit,
        "scaler": scaler,
        "model_obj": model,
        "X_train_final": X_fit,
        "y_train_final": y_fit,
        "X_test_s": X_test_s,
        "y_pred": y_hat,
        "metrik": m
    }

def final_modelleri_olustur(df_ori, df_smogn, X_train_full, y_train_full, X_test, y_test):
    final_modeller = {}

    for model_adi in ["GB", "RF", "ET"]:
        kayit = en_iyi_orijinal_kayit(df_ori, model_adi)
        sonuc = final_modeli_hazirla(kayit, "Orijinal", X_train_full, y_train_full, X_test, y_test)
        if sonuc is not None:
            final_modeller[f"{model_adi} — Orijinal"] = sonuc

    for model_adi in ["GB", "RF", "ET"]:
        adaylar = en_iyi_smogn_kayitlari(df_smogn, model_adi)
        bulundu = False
        for _, kayit in adaylar.head(10).iterrows():
            sonuc = final_modeli_hazirla(kayit, "SMOGN", X_train_full, y_train_full, X_test, y_test)
            if sonuc is not None:
                final_modeller[f"{model_adi} — SMOGN"] = sonuc
                bulundu = True
                print(
                    f"✅ {model_adi} — SMOGN bulundu | "
                    f"k={int(kayit['k'])}, rel_thres={kayit['rel_thres']}, "
                    f"pert={kayit['pert']}, under_samp={kayit['under_samp']}"
                )
                break
        if not bulundu:
            print(f"⚠️ {model_adi} — SMOGN icin calisan final model bulunamadi.")

    return final_modeller

def final_sonuc_tablosu_olustur(final_modeller):
    kayitlar = []
    for isim, paket in final_modeller.items():
        m = paket["metrik"]
        kayitlar.append({
            "Model": isim,
            "Senaryo": paket["Senaryo"],
            "Model_Kisa": paket["Model"],
            "R2_test": round(m["R2"], 5),
            "RMSE_test": round(m["RMSE"], 5),
            "MAE_test": round(m["MAE"], 5),
            "Tail_MAE": round(m["Tail_MAE"], 5),
            "Res_Ort": round(m["Res_Ort"], 5),
            "Res_Std": round(m["Res_Std"], 5),
            "Maks_Hata": round(m["Maks_Hata"], 5),
            "Pm1_kJ_pct": round(m["Pm1_kJ_pct"], 2),
            "Pm2_kJ_pct": round(m["Pm2_kJ_pct"], 2),
            "Genel_Skor": round(genel_skor(m), 5)
        })
    df_final = pd.DataFrame(kayitlar)
    df_final = df_final.sort_values(["R2_test", "RMSE_test"], ascending=[False, True]).reset_index(drop=True)
    return df_final
