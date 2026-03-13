import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR  = "data"
MODEL_DIR = "models"
CSV_PATH  = "module6/window_level_features.csv"

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def run():
    # ── Load CSV from Module 6 ───────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at '{CSV_PATH}'")
        print("Make sure Module 6 has been run first.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")

    # ── Drop NaN rows (windows where TCD was missing) ───────────────────────
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Dropping {nan_count} NaN values across rows...")
    df = df.dropna()
    print(f"Rows after cleaning: {len(df)}")

    # ── Separate features and labels ─────────────────────────────────────────
    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values.astype(np.int32)

    print(f"\nDataset size : {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Before SMOTE — Normal: {np.sum(y==0)}, Tampered: {np.sum(y==1)}")

    # ── Sanity check ─────────────────────────────────────────────────────────
    if len(np.unique(y)) < 2:
        print("\nERROR: Only one class in labels.")
        print("Add normal videos, re-run modules 1–6, then try again.")
        return

    # ── Handle class imbalance with SMOTE ────────────────────────────────────
    n_normal   = np.sum(y == 0)
    n_tampered = np.sum(y == 1)

    if n_normal / (n_tampered + 1e-8) > 1.5:
        smote    = SMOTE(random_state=42)
        X, y     = smote.fit_resample(X, y)
        print(f"After SMOTE  — Normal: {np.sum(y==0)}, Tampered: {np.sum(y==1)}")
    else:
        print("Classes balanced — SMOTE skipped")

    # ── Train / Validation / Test Split (70 / 15 / 15) ──────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes — Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ── Feature Scaling ──────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # fit ONLY on train
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Save everything ──────────────────────────────────────────────────────
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print(f"\n✔ Saved X_train, X_val, X_test, y_train, y_val, y_test → {DATA_DIR}/")
    print(f"✔ Saved scaler.pkl → {MODEL_DIR}/")
    print(f"\nModule 7 complete. Ready for Module 8.")


if __name__ == "__main__":
    run()