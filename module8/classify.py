import numpy as np
import joblib
import os

from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR  = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    required = ["X_train.npy", "X_val.npy", "X_test.npy",
                "y_train.npy", "y_val.npy", "y_test.npy"]

    for f in required:
        if not os.path.exists(os.path.join(DATA_DIR, f)):
            print(f"ERROR: Missing file '{f}' — run Module 7 first.")
            return None

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Combine train + val for final evaluation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print(f"Train+Val : {X_train_full.shape[0]} samples")
    print(f"Test      : {X_test.shape[0]} samples")
    print(f"Features  : {X_test.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


# -----------------------------
# EVALUATE MODEL
# -----------------------------
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name} — Test Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted Normal  Predicted Tampered")
    print(f"  Actual Normal    {cm[0][0]:^15}  {cm[0][1]:^18}")
    print(f"  Actual Tampered  {cm[1][0]:^15}  {cm[1][1]:^18}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Tampered"]))

    return acc, f1


# -----------------------------
# TRAIN SVM
# -----------------------------
def train_svm(X_train, y_train, X_val, y_val):
    print("\n── Training SVM (RBF kernel) ───────────────────────────────────")

    param_grid = {
        "C"    : [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.001],
    }

    svm = GridSearchCV(
        SVC(kernel="rbf", random_state=42, probability=True),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    svm.fit(X_train, y_train)

    print(f"Best SVM params : {svm.best_params_}")
    print(f"Best CV F1 score: {svm.best_score_:.4f}")

    # Validation score
    val_preds = svm.predict(X_val)
    val_f1    = f1_score(y_val, val_preds, average="weighted")
    print(f"Validation F1   : {val_f1:.4f}")

    return svm


# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    print("\n── Training Random Forest ──────────────────────────────────────")

    param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    rf.fit(X_train, y_train)

    print(f"Best RF params  : {rf.best_params_}")
    print(f"Best CV F1 score: {rf.best_score_:.4f}")

    # Validation score
    val_preds = rf.predict(X_val)
    val_f1    = f1_score(y_val, val_preds, average="weighted")
    print(f"Validation F1   : {val_f1:.4f}")

    # Feature importance
    importances = rf.best_estimator_.feature_importances_
    print(f"\n  Feature Importances:")
    for name, score in sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True):
        print(f"    {name:<20} {score:.4f}")

    return rf


# -----------------------------
# MAIN
# -----------------------------
def run():
    print("Loading data...")
    result = load_data()
    if result is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full = result

    # Feature names matching Module 6 output
    feature_names = [
        "TCD", "MeanDelta", "VarDelta",
        "MeanEmb_0", "MeanEmb_1", "MeanEmb_2",
        "StdEmb_0",  "StdEmb_1",  "StdEmb_2"
    ]

    # ── Train both classifiers ───────────────────────────────────────────────
    svm_model = train_svm(X_train, y_train, X_val, y_val)
    rf_model  = train_random_forest(X_train, y_train, X_val, y_val, feature_names)

    # ── Evaluate on test set ─────────────────────────────────────────────────
    svm_acc, svm_f1 = evaluate("SVM",           svm_model, X_test, y_test)
    rf_acc,  rf_f1  = evaluate("Random Forest", rf_model,  X_test, y_test)

    # ── Compare and pick best ────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Final Comparison")
    print(f"{'='*50}")
    print(f"  SVM           — Accuracy: {svm_acc*100:.2f}%  F1: {svm_f1:.4f}")
    print(f"  Random Forest — Accuracy: {rf_acc*100:.2f}%  F1: {rf_f1:.4f}")

    if svm_f1 >= rf_f1:
        best_model      = svm_model
        best_model_name = "SVM"
    else:
        best_model      = rf_model
        best_model_name = "Random Forest"

    print(f"\n  ✔ Best model: {best_model_name}")

    # ── Save both models ─────────────────────────────────────────────────────
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(rf_model,  os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    print(f"\n✔ Saved svm_model.pkl    → {MODEL_DIR}/")
    print(f"✔ Saved rf_model.pkl     → {MODEL_DIR}/")
    print(f"✔ Saved best_model.pkl   → {MODEL_DIR}/")
    print(f"\nModule 8 complete.")


if __name__ == "__main__":
    run()