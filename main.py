"""Command-line entry point for the complete R515B experiment."""

from IPython.display import display

from r515b.config import CIKTI_KLASORU
from r515b.data import iqr_aykiri_raporu, temel_rapor, veriyi_bol, veriyi_yukle
from r515b.modeling import final_modelleri_olustur, final_sonuc_tablosu_olustur, orijinal_tarama, smogn_model_tarama, smogn_parametre_sec
from r515b.visualization import egitim_verisi_dagilim_karsilastirma, final_performans_grafikleri, global_xai_grafikleri, residual_grafikleri, smogn_parametre_grafikleri, sonuclari_excel_kaydet, termodinamik_dependence_grafikleri, veri_analiz_grafikleri

def main():
    from r515b.explainability import detayli_xai, tum_modeller_global_xai
    df_kizgin, df_doymus = veriyi_yukle()
    temel_rapor(df_kizgin, df_doymus)
    df_iqr = iqr_aykiri_raporu(df_kizgin)
    veri_analiz_grafikleri(df_kizgin, df_doymus)

    (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        X_train_full, y_train_full
    ) = veriyi_bol(df_kizgin)

    df_orijinal = orijinal_tarama(X_train, y_train, X_valid, y_valid)

    df_smogn_param_secili, df_smogn_param_ham = smogn_parametre_sec(
        X_train, y_train, X_valid, y_valid
    )
    smogn_parametre_grafikleri(df_smogn_param_ham)

    df_smogn_model = smogn_model_tarama(
        X_train, y_train, X_valid, y_valid, df_smogn_param_secili
    )

    egitim_verisi_dagilim_karsilastirma(X_train_full, y_train_full, df_smogn_model)

    final_modeller = final_modelleri_olustur(
        df_orijinal, df_smogn_model,
        X_train_full, y_train_full, X_test, y_test
    )

    df_final = final_sonuc_tablosu_olustur(final_modeller)
    print("\nFINAL TEST SONUCLARI")
    display(df_final)
    final_performans_grafikleri(df_final)

    df_global, df_tutarlilik, tum_shap, tum_expected = tum_modeller_global_xai(final_modeller, y_test)
    display(df_global)
    display(df_tutarlilik)
    global_xai_grafikleri(df_global)

    detayli_xai(final_modeller, df_final, tum_shap, y_test)
    termodinamik_dependence_grafikleri(final_modeller, tum_shap)

    df_residual = residual_grafikleri(final_modeller, y_test)

    sonuclari_excel_kaydet(
        df_final, df_global, df_tutarlilik,
        df_smogn_param_ham, df_smogn_param_secili, df_smogn_model,
        df_residual, df_iqr
    )

    print("\n✅ Tum analizler tamamlandi.")
    print(f"✅ Ciktilar: {CIKTI_KLASORU}")


def cli():
    """Validate the local input files and optionally run the complete experiment."""
    import argparse
    from r515b.data import check_data

    parser = argparse.ArgumentParser(
        description="R515B enthalpy prediction, SMOGN, and SHAP/LIME analysis"
    )
    parser.add_argument(
        "--check-data", action="store_true",
        help="Validate the input workbooks without fitting the regression models"
    )
    args = parser.parse_args()

    for name, rows, columns in check_data():
        print(f"OK: {name}: {rows} rows, {columns} columns")

    if args.check_data:
        print("Data validation completed. No models were trained.")
    else:
        main()


if __name__ == "__main__":
    cli()
