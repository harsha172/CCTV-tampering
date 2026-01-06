import os
import numpy as np
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
PATCH_FEATURE_FOLDER = "module2/features"          # Module 2 output
QUANTUM_FEATURE_FOLDER = "module5/quantum_features"  # Module 5 output
TCD_FOLDER = "module4/tcd"                         # Module 4 output (assumed npy per window)
MODULE6_FOLDER = "module6"                          # Folder to save CSV
OUTPUT_CSV = os.path.join(MODULE6_FOLDER, "window_level_features.csv")

NUM_QUBITS = 3  # Number of quantum features per patch

# Optional: dictionary mapping window_file -> label
# Example:
# LABELS = {
#   "tampered_window1_features.npy": 1,
#   "normal_window1_features.npy": 0,
# }
LABELS = {}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def compute_mean_std_quantum(quantum_features):
    """Compute mean and std embeddings for a window."""
    mean_emb = quantum_features.mean(axis=0)
    std_emb = quantum_features.std(axis=0)
    return mean_emb, std_emb


def compute_mean_var_delta(patch_features):
    """Compute mean and variance of temporal delta over a window."""
    # patch_features: shape (T, P, F)
    delta = np.abs(patch_features[1:] - patch_features[:-1])  # shape (T-1, P, F)
    mean_delta = delta.mean()
    var_delta = delta.var()
    return mean_delta, var_delta


def load_tcd(window_file):
    """Load TCD for a given window."""
    # Assumes each window has a npy file with single scalar
    tcd_file = os.path.join(TCD_FOLDER, window_file.replace("_features.npy", "_tcd.npy"))
    if os.path.exists(tcd_file):
        return float(np.load(tcd_file))
    else:
        # If missing, return NaN
        return np.nan


# -----------------------------
# MAIN PROCESS
# -----------------------------
# Ensure module6 folder exists
if not os.path.exists(MODULE6_FOLDER):
    os.makedirs(MODULE6_FOLDER)

dataset = []
window_files = sorted([f for f in os.listdir(PATCH_FEATURE_FOLDER) if f.endswith(".npy")])

for window_file in window_files:
    print(f"Processing window: {window_file}")

    # Load patch features (Module 2)
    patch_features = np.load(os.path.join(PATCH_FEATURE_FOLDER, window_file))  # shape (T, P, F)

    # Aggregate over frames if needed
    if patch_features.ndim == 3:
        patch_features_agg = patch_features.mean(axis=0)  # shape (P, F)
    else:
        patch_features_agg = patch_features  # already (P, F)

    # Load quantum embeddings (Module 5)
    quantum_features = np.load(os.path.join(QUANTUM_FEATURE_FOLDER, window_file))  # shape (P, Q)

    # Compute mean & std of quantum embeddings
    mean_emb, std_emb = compute_mean_std_quantum(quantum_features)  # each shape (Q,)

    # Compute mean & var of temporal delta (Module 2 features)
    mean_delta, var_delta = compute_mean_var_delta(patch_features)  # scalars

    # Load TCD (Module 4)
    tcd_value = load_tcd(window_file)  # scalar

    # Combine into one window feature vector
    window_vector = np.concatenate([
        [tcd_value, mean_delta, var_delta],
        mean_emb,
        std_emb
    ])

    # Append label if available
    label = LABELS.get(window_file, 0)  # default 0
    window_vector_with_label = np.concatenate([window_vector, [label]])

    dataset.append(window_vector_with_label)

# -----------------------------
# SAVE TO CSV
# -----------------------------
Q = NUM_QUBITS
columns = ['TCD', 'MeanDelta', 'VarDelta']
columns += [f'MeanEmb_{i}' for i in range(Q)]
columns += [f'StdEmb_{i}' for i in range(Q)]
columns += ['Label']

df = pd.DataFrame(dataset, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Window-level feature CSV saved: {OUTPUT_CSV}")
