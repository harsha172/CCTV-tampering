import os
import numpy as np
from itertools import combinations

# -----------------------------
# CONFIGURATION
# -----------------------------
PATCH_FEATURE_FOLDER = "module2/features"
TCD_OUTPUT_FOLDER = "module4/tcd"

lambda_weight = 0.3


# -----------------------------
# LOAD PATCH FEATURES
# -----------------------------
def load_patch_features(folder_path):
    """
    Loads all window feature files from a video folder.
    Returns list of np.ndarray, each shape (P, F)
    """
    feature_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ])

    patch_features = []
    filenames = []
    for file in feature_files:
        file_path = os.path.join(folder_path, file)
        features = np.load(file_path)

        # If shape is (T, P, F) → average over time axis → (P, F)
        if features.ndim == 3:
            features = features.mean(axis=0)

        patch_features.append(features)
        filenames.append(file)

    return patch_features, filenames


# -----------------------------
# TCD COMPUTATION
# -----------------------------
def compute_temporal_delta(curr_feats, prev_feats):
    delta = curr_feats - prev_feats
    return np.linalg.norm(delta, axis=1)


def compute_pairwise_distances(patch_feats):
    distances = []
    for i, j in combinations(range(len(patch_feats)), 2):
        d = np.linalg.norm(patch_feats[i] - patch_feats[j])
        distances.append(d)
    return np.array(distances)


def compute_tcd_window(curr_feats, prev_feats, lambda_weight=0.5):
    if curr_feats.shape[0] < 2:
        return np.nan

    curr_feats = np.nan_to_num(curr_feats)
    prev_feats = np.nan_to_num(prev_feats)

    delta_vals = compute_temporal_delta(curr_feats, prev_feats)
    mean_delta = np.mean(np.abs(delta_vals))

    pairwise_dists = compute_pairwise_distances(curr_feats)
    var_dist = np.var(pairwise_dists) if len(pairwise_dists) > 0 else 0.0

    return var_dist + lambda_weight * mean_delta


def compute_tcd_series(patch_features, lambda_weight=0.5):
    """Returns dict: window_index → TCD score"""
    tcd_scores = {}
    for t in range(1, len(patch_features)):
        tcd_scores[t] = compute_tcd_window(
            patch_features[t],
            patch_features[t - 1],
            lambda_weight
        )
    return tcd_scores


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(TCD_OUTPUT_FOLDER, exist_ok=True)

    frame_folders = sorted([
        f for f in os.listdir(PATCH_FEATURE_FOLDER)
        if os.path.isdir(os.path.join(PATCH_FEATURE_FOLDER, f))
    ])

    if not frame_folders:
        print("No video folders found in module2/features/")
        return

    for frame in frame_folders:
        frame_path = os.path.join(PATCH_FEATURE_FOLDER, frame)
        patch_features, filenames = load_patch_features(frame_path)

        if len(patch_features) < 2:
            print(f"Skipping {frame} — not enough windows (need at least 2)")
            continue

        tcd_scores = compute_tcd_series(patch_features, lambda_weight)

        # ── Save TCD scores to disk ──────────────────────────────────────────
        # Module 6 will load these, so we match the expected filename format:
        # window{t}_features_tcd.npy
        frame_out = os.path.join(TCD_OUTPUT_FOLDER, frame)
        os.makedirs(frame_out, exist_ok=True)

        for t, score in tcd_scores.items():
            # Match the source window filename, just swap suffix
            source_filename = filenames[t]  # e.g. "window1_features.npy"
            tcd_filename = source_filename.replace(".npy", "_tcd.npy")
            out_path = os.path.join(frame_out, tcd_filename)
            np.save(out_path, np.array(score, dtype=np.float32))

        print(f"[{frame}] Saved {len(tcd_scores)} TCD scores → {frame_out}")

        # Print summary
        for t, score in tcd_scores.items():
            print(f"   Window {t} → TCD = {score:.5f}")


if __name__ == "__main__":
    main()