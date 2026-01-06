import os
import numpy as np
from itertools import combinations

PATCH_FEATURE_FOLDER = "module2/features"

def load_patch_features(folder_path):
    """
    Loads patch features saved per window.

    Args:
        folder_path (str): path to module2/features

    Returns:
        list of np.ndarray: index = window t, value = (P, F)
    """
    feature_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".npy")
    ])

    patch_features = []

    for file in feature_files:
        file_path = os.path.join(folder_path, file)
        features = np.load(file_path)
        patch_features.append(features)

    return patch_features

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
    # Temporal change
    delta_vals = compute_temporal_delta(curr_feats, prev_feats)
    mean_delta = np.mean(np.abs(delta_vals))

    # Spatial inconsistency
    pairwise_dists = compute_pairwise_distances(curr_feats)
    var_dist = np.var(pairwise_dists)

    # Final TCD
    tcd = var_dist + lambda_weight * mean_delta
    return tcd

def compute_tcd_series(patch_features, lambda_weight=0.5):
    tcd_scores = {}

    for t in range(1, len(patch_features)):
        tcd_scores[t] = compute_tcd_window(
            patch_features[t],
            patch_features[t - 1],
            lambda_weight
        )

    return tcd_scores

patch_features = load_patch_features(PATCH_FEATURE_FOLDER)

print("Total windows:", len(patch_features))
print("Shape of one window:", patch_features[0].shape)

lambda_weight = 0.3
tcd_scores = compute_tcd_series(patch_features, lambda_weight)

for t, score in tcd_scores.items():
    print(f"Window {t} → TCD = {score:.5f}")
