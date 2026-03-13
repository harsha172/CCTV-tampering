import os
import numpy as np
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
PATCH_FEATURE_FOLDER   = "module2/features"
QUANTUM_FEATURE_FOLDER = "module5/quantum_features"
TCD_FOLDER             = "module4/tcd"
MODULE6_FOLDER         = "module6"
OUTPUT_CSV             = os.path.join(MODULE6_FOLDER, "window_level_features.csv")

NUM_QUBITS = 3

# -----------------------------
# LABELS
# 0 = normal/authentic
# 1 = tampered/fake
# Add more entries here if you add more videos later
# -----------------------------
LABELS = {
    "tampered": 1,
    "normal1":  0,
    "normal2":  0,
    "normal3":  0,
    "normal4":  0,
    "normal5":  0,
    "normal6":  0,
    "normal7":  0,
    "normal8":  0,
    "normal9":  0,
    "normal10":  0,
    "normal11":  0,
    "normal12":  0,
    "normal13":  0,
    "normal14":  0,
    "normal15":  0,
    "normal16":  0,
    "normal17":  0,
    "normal18":  0,
    "normal19":  0,
    "tampered2":  1,
    "tampered3":  1,
    "tampered4":  1,
    "tampered5":  1,
    "tampered6":  1,
    "tampered7":  1,
}


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_mean_std_quantum(quantum_features):
    """Returns mean and std of quantum embeddings across patches."""
    mean_emb = quantum_features.mean(axis=0)
    std_emb  = quantum_features.std(axis=0)
    return mean_emb, std_emb


def compute_mean_var_delta(patch_features):
    """Returns mean and variance of temporal change across frames."""
    if patch_features.shape[0] < 2:
        return 0.0, 0.0
    delta = np.abs(patch_features[1:] - patch_features[:-1])
    return float(delta.mean()), float(delta.var())


def load_tcd(frame, window_file):
    """
    Loads the TCD score saved by Module 4.
    Expects: module4/tcd/<frame>/<window_file replacing .npy with _tcd.npy>
    """
    tcd_filename = window_file.replace(".npy", "_tcd.npy")
    tcd_path = os.path.join(TCD_FOLDER, frame, tcd_filename)

    if os.path.exists(tcd_path):
        return float(np.load(tcd_path))
    else:
        return np.nan


# -----------------------------
# MAIN PROCESS
# -----------------------------
def main():
    os.makedirs(MODULE6_FOLDER, exist_ok=True)

    dataset = []
    missing_tcd_count    = 0
    missing_quantum_count = 0

    frame_folders = sorted([
        f for f in os.listdir(PATCH_FEATURE_FOLDER)
        if os.path.isdir(os.path.join(PATCH_FEATURE_FOLDER, f))
    ])

    if not frame_folders:
        print("No video folders found in module2/features/")
        return

    for frame in frame_folders:
        print(f"\nProcessing: {frame}")

        patch_frame_path   = os.path.join(PATCH_FEATURE_FOLDER, frame)
        quantum_frame_path = os.path.join(QUANTUM_FEATURE_FOLDER, frame)

        # Get label for this video
        if frame not in LABELS:
            print(f"   ⚠ '{frame}' not in LABELS dict — skipping this video")
            continue

        label = LABELS[frame]

        if not os.path.exists(quantum_frame_path):
            print(f"   ✗ Skipping — missing quantum features folder")
            continue

        window_files = sorted([
            f for f in os.listdir(patch_frame_path)
            if f.endswith(".npy")
        ])

        for window_file in window_files:

            # ── Module 2: patch features ─────────────────────────────────────
            patch_path     = os.path.join(patch_frame_path, window_file)
            patch_features = np.load(patch_path)
            patch_features = np.nan_to_num(patch_features)

            # Ensure shape is (T, P, F) for delta computation
            if patch_features.ndim == 2:
                patch_features = patch_features[np.newaxis, ...]  # → (1, P, F)

            # ── Module 5: quantum features ───────────────────────────────────
            quantum_path = os.path.join(quantum_frame_path, window_file)
            if not os.path.exists(quantum_path):
                missing_quantum_count += 1
                print(f"   ✗ Missing quantum file: {window_file}")
                continue

            quantum_features = np.load(quantum_path)
            if quantum_features.shape[0] == 0:
                continue

            # ── Module 4: TCD score ──────────────────────────────────────────
            tcd_value = load_tcd(frame, window_file)
            if np.isnan(tcd_value):
                missing_tcd_count += 1

            # ── Combine into one feature vector ─────────────────────────────
            mean_emb, std_emb     = compute_mean_std_quantum(quantum_features)
            mean_delta, var_delta = compute_mean_var_delta(patch_features)

            window_vector = np.concatenate([
                [tcd_value, mean_delta, var_delta],  # 3 features
                mean_emb,                             # NUM_QUBITS features
                std_emb,                              # NUM_QUBITS features
                [label]                               # label last
            ])

            dataset.append(window_vector)

        print(f"   ✔ {len(window_files)} windows processed, label={label}")

    # ── Build and save CSV ───────────────────────────────────────────────────
    if not dataset:
        print("\nNo data collected — check your folder paths and that modules 2, 4, 5 have run.")
        return

    Q       = NUM_QUBITS
    columns  = ['TCD', 'MeanDelta', 'VarDelta']
    columns += [f'MeanEmb_{i}' for i in range(Q)]
    columns += [f'StdEmb_{i}'  for i in range(Q)]
    columns += ['Label']

    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"Total windows saved  : {len(df)}")
    print(f"Label distribution   :\n{df['Label'].value_counts().to_string()}")
    print(f"Missing TCD files    : {missing_tcd_count}")
    print(f"Missing quantum files: {missing_quantum_count}")
    print(f"CSV saved at         : {OUTPUT_CSV}")

    if df['Label'].nunique() == 1:
        print("\n⚠  WARNING: All labels are the same value!")
        print("   Add normal videos and re-run the full pipeline before running Module 7.")


if __name__ == "__main__":
    main()