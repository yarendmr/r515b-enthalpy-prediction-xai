"""Input workbooks, data quality, transformation, and splitting."""

from sklearn.preprocessing import StandardScaler
from IPython.display import display
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DOYMUS_YOL, HEDEF, KIZGIN_YOL, OZELLIKLER, SEED

def scaler_fit_transform(X_train_df, X_other_df):
    scaler = StandardScaler()
    scaler.fit(X_train_df)
    return scaler, scaler.transform(X_train_df), scaler.transform(X_other_df)

def veriyi_yukle():
    df_kizgin = pd.read_excel(KIZGIN_YOL, skiprows=1)
    df_doymus = pd.read_excel(DOYMUS_YOL, skiprows=1)

    df_kizgin.columns = ["T", "P", "v", "h", "s", "durum"]
    df_doymus.columns = ["T", "P", "v_sivi", "v_buhar", "h_sivi", "h_buhar", "s_sivi", "s_buhar"]

    if "durum" in df_kizgin.columns:
        df_kizgin = df_kizgin.drop(columns=["durum"])

    for col in OZELLIKLER + [HEDEF]:
        df_kizgin[col] = pd.to_numeric(df_kizgin[col], errors="coerce")

    df_kizgin = df_kizgin.dropna(subset=OZELLIKLER + [HEDEF]).reset_index(drop=True)
    return df_kizgin, df_doymus

def temel_rapor(df_kizgin, df_doymus):
    print("=" * 60)
    print("TEMEL VERI KONTROLU")
    print("=" * 60)
    print(f"Doymus veri : {df_doymus.shape[0]} satir, {df_doymus.shape[1]} sutun")
    print(f"Kizgin veri : {df_kizgin.shape[0]} satir, {df_kizgin.shape[1]} sutun")
    for isim, df in [("Doymus", df_doymus), ("Kizgin", df_kizgin)]:
        print(f"\n--- {isim} ---")
        print("Eksik deger :", int(df.isnull().sum().sum()))
        print("Tekrar satir:", int(df.duplicated().sum()))
    print("\nKizgin veri istatistikleri:")
    display(df_kizgin.describe().round(4))
    print("\nKizgin veri korelasyon matrisi:")
    display(df_kizgin[["T", "P", "v", "h", "s"]].corr().round(3))

def iqr_aykiri_raporu(df):
    satirlar = []
    print("\nIQR aykiri deger analizi (silinmiyor):")
    for col in OZELLIKLER + [HEDEF]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"  {col:4s}: {int(n_out)}")
        satirlar.append({"Degisken": col, "IQR_Aykiri_Sayi": int(n_out)})
    return pd.DataFrame(satirlar)

def veriyi_bol(df):
    X = df[OZELLIKLER].copy()
    y = df[HEDEF].copy()
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=SEED)
    print("\nVeri bolme:")
    print(f"Train      : {len(X_train)}")
    print(f"Validation : {len(X_valid)}")
    print(f"Test       : {len(X_test)}")
    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_train_full, y_train_full


# Raw source workbooks are local inputs; these validation rules do not mutate them.
from pathlib import Path
from .config import PROJE_KOKU

RAW_DATA = PROJE_KOKU / "data" / "raw"
FILES = {
    "Superheated vapor": (RAW_DATA / "R515B_kizgin_buhar.xlsx", 6, 578),
    "Saturated properties": (RAW_DATA / "R515B_doymus_ozellikleri.xlsx", 8, 136),
}

def check_data():
    """Return (name, row_count, clean_column_count) tuples; raise on invalid inputs."""
    findings = []
    for name, (path, expected_raw_columns, expected_rows) in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input: {path}\n"
                "Place both Excel files in data/raw/; see README.md."
            )
        df = pd.read_excel(path, skiprows=1)
        if df.shape != (expected_rows, expected_raw_columns):
            raise ValueError(
                f"Unexpected input shape for {name}: {df.shape}; "
                f"expected {(expected_rows, expected_raw_columns)}."
            )
        values = df.iloc[:, :5] if expected_raw_columns == 6 else df
        if values.isna().any().any():
            raise ValueError(f"Missing values in {name}: {path}")
        if df.duplicated().any():
            raise ValueError(f"Duplicate rows in {name}: {path}")
        if expected_raw_columns == 6:
            for col in range(5):
                if pd.to_numeric(df.iloc[:, col], errors="coerce").isna().any():
                    raise ValueError(f"Non-numeric values in {name} column {col + 1}")
        findings.append((name, len(df), 5 if expected_raw_columns == 6 else 8))
    return findings
