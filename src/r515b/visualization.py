"""Figures and Excel result exports."""

import matplotlib
matplotlib.use("Agg")  # Export figures without an interactive window.

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.0)

from .config import CIKTI_KLASORU, GRI, HEDEF, KIZIL, MAVI, MOR, OZELLIKLER, TURUNCU, YESIL
from .resampling import smogn_veri_uret

def fig_kaydet(fig_adi):
    yol = os.path.join(CIKTI_KLASORU, fig_adi)
    plt.savefig(yol, dpi=150, bbox_inches="tight")

def veri_analiz_grafikleri(df_kizgin, df_doymus):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(df_doymus["T"], df_doymus["P"], color=MAVI, linewidth=2)
    axes[0, 0].set_title("Doymus Faz: T-P Iliskisi")
    axes[0, 0].set_xlabel("T")
    axes[0, 0].set_ylabel("P")

    axes[0, 1].plot(df_doymus["T"], df_doymus["h_sivi"], label="h_sivi", color=YESIL)
    axes[0, 1].plot(df_doymus["T"], df_doymus["h_buhar"], label="h_buhar", color=KIZIL)
    axes[0, 1].set_title("Doymus Faz: Entalpi")
    axes[0, 1].legend()

    axes[1, 0].plot(df_doymus["T"], df_doymus["s_sivi"], label="s_sivi", color=YESIL)
    axes[1, 0].plot(df_doymus["T"], df_doymus["s_buhar"], label="s_buhar", color=KIZIL)
    axes[1, 0].set_title("Doymus Faz: Entropi")
    axes[1, 0].legend()

    axes[1, 1].semilogy(df_doymus["T"], df_doymus["v_sivi"], label="v_sivi", color=YESIL)
    axes[1, 1].semilogy(df_doymus["T"], df_doymus["v_buhar"], label="v_buhar", color=KIZIL)
    axes[1, 1].set_title("Doymus Faz: Ozgul Hacim")
    axes[1, 1].legend()

    plt.tight_layout()
    fig_kaydet("sekil_01_doymus_faz_genel.png")
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, feat, renk in zip(axes, ["T", "P", "v", "h", "s"], [MAVI, TURUNCU, YESIL, KIZIL, MOR]):
        ax.hist(df_kizgin[feat], bins=25, color=renk, edgecolor="white", alpha=0.85)
        ax.set_title(feat)
    plt.tight_layout()
    fig_kaydet("sekil_02_kizgin_dagilimlar.png")
    plt.show()
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, feat in zip(axes, ["T", "P", "v", "s"]):
        ax.scatter(df_kizgin[feat], df_kizgin["h"], alpha=0.45, s=18, color=MAVI)
        ax.set_title(f"{feat} - h")
        ax.set_xlabel(feat)
        ax.set_ylabel("h")
    plt.tight_layout()
    fig_kaydet("sekil_03_girdi_hedef_iliskileri.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.heatmap(df_kizgin[["T", "P", "v", "h", "s"]].corr(), annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Korelasyon Matrisi")
    plt.tight_layout()
    fig_kaydet("sekil_04_korelasyon_matrisi.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(df_kizgin["h"], bins=25, kde=True, color=KIZIL)
    plt.title("Hedef Degisken h Dagilimi")
    plt.xlabel("h")
    plt.tight_layout()
    fig_kaydet("sekil_05_h_dagilimi.png")
    plt.show()
    plt.close()

def smogn_parametre_grafikleri(df_smogn_param):
    if len(df_smogn_param) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sirali = df_smogn_param.sort_values("Tail_MAE_valid").head(15).copy()
    etiketler = [
        f"k={r['k']}\nth={r['rel_thres']}\np={r['pert']}\nu={int(r['under_samp'])}"
        for _, r in sirali.iterrows()
    ]
    axes[0].bar(range(len(sirali)), sirali["R2_valid"], color=MAVI, alpha=0.85)
    axes[0].set_title("SMOGN Parametreleri - R2")
    axes[0].set_xticks(range(len(sirali)))
    axes[0].set_xticklabels(etiketler, rotation=45, ha="right", fontsize=7)

    axes[1].bar(range(len(sirali)), sirali["RMSE_valid"], color=KIZIL, alpha=0.85)
    axes[1].set_title("SMOGN Parametreleri - RMSE")
    axes[1].set_xticks(range(len(sirali)))
    axes[1].set_xticklabels(etiketler, rotation=45, ha="right", fontsize=7)

    axes[2].bar(range(len(sirali)), sirali["Tail_MAE_valid"], color=YESIL, alpha=0.85)
    axes[2].set_title("SMOGN Parametreleri - Tail MAE")
    axes[2].set_xticks(range(len(sirali)))
    axes[2].set_xticklabels(etiketler, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    fig_kaydet("sekil_06_smogn_parametre_karsilastirma.png")
    plt.show()
    plt.close()

def egitim_verisi_dagilim_karsilastirma(X_train_full, y_train_full, df_smogn_model):
    if len(df_smogn_model) == 0:
        return
    df_train_full_birlesik = pd.concat([X_train_full.reset_index(drop=True), y_train_full.reset_index(drop=True)], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, model_adi, renk in zip(axes, ["GB", "RF", "ET"], [KIZIL, MAVI, YESIL]):
        alt = df_smogn_model[df_smogn_model["Model"] == model_adi]
        if len(alt) == 0:
            continue
        sec = alt.sort_values(["Tail_MAE_valid", "MAE_valid", "RMSE_valid"], ascending=[True, True, True]).iloc[0]
        df_aug = smogn_veri_uret(
            df_train_full_birlesik, HEDEF,
            int(sec["k"]), float(sec["rel_thres"]), float(sec["pert"]), bool(sec["under_samp"])
        )
        if df_aug is None:
            continue
        ax.hist(y_train_full.values, bins=25, alpha=0.55, color=GRI, label=f"Orijinal (n={len(y_train_full)})", edgecolor="white")
        ax.hist(df_aug[HEDEF].values, bins=25, alpha=0.55, color=renk, label=f"SMOGN (n={len(df_aug)})", edgecolor="white")
        ax.set_title(f"{model_adi} icin h dagilimi")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig_kaydet("sekil_07_orijinal_smogn_dagilim_karsilastirma.png")
    plt.show()
    plt.close()

def final_performans_grafikleri(df_final):
    if len(df_final) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metrik, baslik, renk in zip(
        axes,
        ["R2_test", "RMSE_test", "MAE_test"],
        ["R2", "RMSE", "MAE"],
        [MAVI, KIZIL, YESIL]
    ):
        bars = ax.bar(df_final["Model"], df_final[metrik], color=renk, alpha=0.85, edgecolor="white")
        ax.set_title(f"Final Test - {baslik}")
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, df_final[metrik]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig_kaydet("sekil_08_final_model_performans_karsilastirma.png")
    plt.show()
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = axes[0].bar(df_final["Model"], df_final["Tail_MAE"], color=TURUNCU, alpha=0.85, edgecolor="white")
    axes[0].set_title("Tail MAE")
    axes[0].tick_params(axis="x", rotation=25)
    for b, v in zip(bars1, df_final["Tail_MAE"]):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    bars2 = axes[1].bar(df_final["Model"], df_final["Genel_Skor"], color=MOR, alpha=0.85, edgecolor="white")
    axes[1].set_title("Genel Skor")
    axes[1].tick_params(axis="x", rotation=25)
    for b, v in zip(bars2, df_final["Genel_Skor"]):
        axes[1].text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_kaydet("sekil_09_tailmae_genelskor.png")
    plt.show()
    plt.close()

def residual_grafikleri(final_modeller, y_test):
    kayitlar = []
    for model_adi, paket in final_modeller.items():
        y_pred = paket["y_pred"]
        for i in range(len(y_test)):
            res = float(np.array(y_test)[i]) - float(y_pred[i])
            kayitlar.append({
                "Model": model_adi,
                "Gercek_h": float(np.array(y_test)[i]),
                "Tahmin_h": float(y_pred[i]),
                "Residual": res,
                "Abs_Hata": abs(res)
            })
    df_residual = pd.DataFrame(kayitlar)

    plt.figure(figsize=(11, 5))
    sns.histplot(data=df_residual, x="Residual", hue="Model", kde=True, element="step")
    plt.title("Residual Dagilimlari")
    plt.tight_layout()
    fig_kaydet("sekil_10_residual_hist.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.scatterplot(data=df_residual, x="Tahmin_h", y="Residual", hue="Model", alpha=0.6)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Tahmin - Residual")
    plt.tight_layout()
    fig_kaydet("sekil_11_tahmin_residual.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.scatterplot(data=df_residual, x="Gercek_h", y="Residual", hue="Model", alpha=0.6)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.title("Gercek - Residual")
    plt.tight_layout()
    fig_kaydet("sekil_12_gercek_residual.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.scatterplot(data=df_residual, x="Gercek_h", y="Tahmin_h", hue="Model", alpha=0.65)
    alt = min(df_residual["Gercek_h"].min(), df_residual["Tahmin_h"].min())
    ust = max(df_residual["Gercek_h"].max(), df_residual["Tahmin_h"].max())
    plt.plot([alt, ust], [alt, ust], "k--", linewidth=1.2)
    plt.title("Gercek - Tahmin Karsilastirmasi")
    plt.tight_layout()
    fig_kaydet("sekil_13_gercek_tahmin.png")
    plt.show()
    plt.close()

    return df_residual

def global_xai_grafikleri(df_global):
    if len(df_global) == 0:
        return
    for pivot_col, baslik, fig_adi in [
        ("Mean_Abs_SHAP", "Tum Modeller icin Mean |SHAP|", "sekil_14_mean_abs_shap.png"),
        ("FI", "Tum Modeller icin Feature Importance", "sekil_15_feature_importance.png"),
        ("Perm_Mean", "Tum Modeller icin Permutation Importance", "sekil_16_permutation_importance.png")
    ]:
        pivot = df_global.pivot(index="Ozellik", columns="Model", values=pivot_col)
        pivot.plot(kind="bar", figsize=(13, 6))
        plt.title(baslik)
        plt.ylabel(pivot_col)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig_kaydet(fig_adi)
        plt.show()
        plt.close()

def termodinamik_dependence_grafikleri(final_modeller, tum_shap):
    for model_adi, paket in final_modeller.items():
        if model_adi not in tum_shap:
            continue
        shap_vals = tum_shap[model_adi]
        X_test_s = paket["X_test_s"]
        X_org = X_test_s * paket["scaler"].scale_ + paket["scaler"].mean_

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(f"SHAP Degerleri ve Termodinamik Tutarlilik — {model_adi}", fontsize=13, fontweight="bold")
        birimler = {"T": "°C", "P": "kPa", "v": "m³/kg", "s": "kJ/kg·K"}
        renk_feat = {"T": KIZIL, "P": YESIL, "v": TURUNCU, "s": MAVI}
        for ax, feat in zip(axes.flat, OZELLIKLER):
            fi = OZELLIKLER.index(feat)
            ax.scatter(X_org[:, fi], shap_vals[:, fi], color=renk_feat[feat], alpha=0.45, s=14)
            z = np.polyfit(X_org[:, fi], shap_vals[:, fi], 1)
            xs = np.linspace(X_org[:, fi].min(), X_org[:, fi].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs), "k-", lw=2.2, label=f"Eğim: {z[0]:+.3f}")
            ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.5)
            ax.set_title(feat)
            ax.set_xlabel(f"{feat} ({birimler[feat]})")
            ax.set_ylabel(f"SHAP({feat})")
            ax.legend(fontsize=9)
        plt.tight_layout()
        fig_kaydet(f"sekil_termodinamik_{model_adi.replace(' ', '_').replace('—', '-')}.png")
        plt.show()
        plt.close()

def sonuclari_excel_kaydet(df_final, df_global, df_tutarlilik, df_smogn_param_ham, df_smogn_param_secili, df_smogn_model, df_residual, df_iqr):
    excel_yolu = os.path.join(CIKTI_KLASORU, "R515B_final_smogn_guclu_raporlar.xlsx")
    with pd.ExcelWriter(excel_yolu, engine="openpyxl") as writer:
        df_final.to_excel(writer, sheet_name="Model_Performanslari", index=False)
        df_global.to_excel(writer, sheet_name="Global_XAI_Metrikleri", index=False)
        df_tutarlilik.to_excel(writer, sheet_name="Fiziksel_Tutarlilik", index=False)
        df_smogn_param_ham.to_excel(writer, sheet_name="SMOGN_Param_Ham", index=False)
        df_smogn_param_secili.to_excel(writer, sheet_name="SMOGN_Param_Secili", index=False)
        df_smogn_model.to_excel(writer, sheet_name="SMOGN_Model_Arama", index=False)
        df_residual.to_excel(writer, sheet_name="Residual_Detay", index=False)
        df_iqr.to_excel(writer, sheet_name="IQR_Aykiri_Ozeti", index=False)
    print(f"\n✅ Excel kaydedildi: {excel_yolu}")
