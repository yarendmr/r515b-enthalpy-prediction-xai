"""Local and global SHAP/LIME explanations and trend analyses."""

from lime import lime_tabular
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap

from .config import HEDEF, KIZIL, MAVI, OZELLIKLER, SEED
from .visualization import fig_kaydet

def yon_yorumla(feat, egim):
    beklenen = {"T": "+", "P": "-", "v": "+", "s": "+"}
    gozlenen = "+" if egim > 0 else "-"
    if feat in ["T", "s"]:
        if gozlenen == beklenen[feat]:
            return gozlenen, "Guclu uyum", "Beklenen yonle uyumlu"
        return gozlenen, "Uyumsuz", "Beklenen yonle celisiyor"
    if feat == "P":
        if egim < -0.001:
            return gozlenen, "Uyumlu", "Negatif yon gozleniyor"
        if egim <= 0.001:
            return gozlenen, "Zayif ama uyumlu", "Etki cok zayif"
        return gozlenen, "Uyumsuz", "Pozitif egilim dikkatle yorumlanmali"
    if feat == "v":
        if egim > 0.05:
            return gozlenen, "Uyumlu", "Pozitif yon gozleniyor"
        if -0.05 <= egim <= 0.05:
            return gozlenen, "Belirsiz", "Yon net degil"
        return gozlenen, "Uyumsuz", "Negatif egilim dikkatle yorumlanmali"
    return gozlenen, "Belirsiz", "Yorumlanamadi"

def idx_entalpi(y_dizi):
    arr = np.array(y_dizi)
    return int(np.argmin(arr)), int(np.argsort(arr)[len(arr)//2]), int(np.argmax(arr))

def tum_modeller_global_xai(final_modeller, y_test):
    tum_global = []
    tum_tutarlilik = []
    tum_shap = {}
    tum_expected = {}

    for model_adi, paket in final_modeller.items():
        print(f"\n{model_adi} XAI isleniyor...")
        model = paket["model_obj"]
        X_test_s = paket["X_test_s"]

        perm = permutation_importance(model, X_test_s, y_test, scoring="r2", n_repeats=15, random_state=SEED)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_s)
        expected_val = float(explainer.expected_value)

        tum_shap[model_adi] = shap_vals
        tum_expected[model_adi] = expected_val

        fi = model.feature_importances_ if hasattr(model, "feature_importances_") else np.zeros(len(OZELLIKLER))
        X_org = X_test_s * paket["scaler"].scale_ + paket["scaler"].mean_

        for i, feat in enumerate(OZELLIKLER):
            egim = np.polyfit(X_org[:, i], shap_vals[:, i], 1)[0]
            gy, duzey, yorum = yon_yorumla(feat, egim)

            tum_global.append({
                "Model": model_adi,
                "Ozellik": feat,
                "FI": round(float(fi[i]), 5),
                "Perm_Mean": round(float(perm.importances_mean[i]), 5),
                "Mean_Abs_SHAP": round(float(np.abs(shap_vals[:, i]).mean()), 5),
                "SHAP_Egimi": round(float(egim), 5)
            })

            tum_tutarlilik.append({
                "Model": model_adi,
                "Ozellik": feat,
                "SHAP_Egimi": round(float(egim), 5),
                "Beklenen_Yon": {"T": "+", "P": "-", "v": "+", "s": "+"}[feat],
                "Gozlenen_Yon": gy,
                "Tutar_Duzeyi": duzey,
                "Yorum": yorum
            })

    return pd.DataFrame(tum_global), pd.DataFrame(tum_tutarlilik), tum_shap, tum_expected

def detayli_xai(final_modeller, df_final, tum_shap, y_test):
    smogn_df = df_final[df_final["Senaryo"] == "SMOGN"]
    secilenler = [df_final.iloc[0]["Model"]]

    if len(smogn_df) > 0 and smogn_df.iloc[0]["Model"] not in secilenler:
        secilenler.append(smogn_df.iloc[0]["Model"])

    for model_adi in secilenler:
        paket = final_modeller[model_adi]
        model = paket["model_obj"]
        X_test_s = paket["X_test_s"]
        y_pred = paket["y_pred"]
        shap_vals = tum_shap[model_adi]

        idx_low, idx_mid, idx_high = idx_entalpi(y_test)
        ornekler = [("Dusuk h", idx_low), ("Medyan h", idx_mid), ("Yuksek h", idx_high)]

        plt.figure(figsize=(7, 4))
        mean_abs = np.abs(shap_vals).mean(axis=0)
        sirali = np.argsort(mean_abs)
        plt.barh([OZELLIKLER[i] for i in sirali], mean_abs[sirali], color=KIZIL if "GB" in model_adi else MAVI, alpha=0.85)
        plt.title(f"SHAP Kuresel Ozellik Onemi — {model_adi}")
        plt.xlabel("Ortalama |SHAP|")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig_kaydet(f"shap_bar_{model_adi.replace(' ', '_').replace('—', '-')}.png")
        plt.show()
        plt.close()

        shap.summary_plot(shap_vals, X_test_s, feature_names=OZELLIKLER, show=False, plot_size=(10, 5))
        plt.title(f"SHAP Beeswarm — {model_adi}")
        plt.tight_layout()
        fig_kaydet(f"shap_beeswarm_{model_adi.replace(' ', '_').replace('—', '-')}.png")
        plt.show()
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"SHAP Yerel Aciklamalar — {model_adi}", fontweight="bold")
        for ax, (baslik, idx) in zip(axes, ornekler):
            katkilar = shap_vals[idx]
            sir = np.argsort(np.abs(katkilar))
            etiketler = [OZELLIKLER[i] for i in sir]
            degerler = [katkilar[i] for i in sir]
            ax.barh(etiketler, degerler, color=[KIZIL if x > 0 else MAVI for x in degerler], alpha=0.85)
            ax.axvline(0, color="black", linewidth=1.2)
            ax.set_title(f"{baslik} (Gercek: {y_test.iloc[idx]:.1f}, Tahmin: {y_pred[idx]:.1f})")
            ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig_kaydet(f"shap_yerel_{model_adi.replace(' ', '_').replace('—', '-')}.png")
        plt.show()
        plt.close()

        print(f"\n--- LIME Yerel Aciklamalar ({model_adi}) ---")
        explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=np.array(paket["X_train_final"]),
            feature_names=OZELLIKLER,
            class_names=[HEDEF],
            mode="regression",
            random_state=SEED
        )

        for baslik, idx in ornekler:
            X_org_instance = X_test_s[idx] * paket["scaler"].scale_ + paket["scaler"].mean_
            exp = explainer_lime.explain_instance(
                data_row=X_org_instance,
                predict_fn=lambda x: model.predict(paket["scaler"].transform(x))
            )
            print(f"\n[{baslik}] Tahmin katkisi:")
            for s, agirlik in exp.as_list():
                print(f"  {s:30s} : {agirlik:+.4f}")
