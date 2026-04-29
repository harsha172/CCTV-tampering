import os
import sys
import cv2
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
from itertools import combinations
from skimage.measure import shannon_entropy
import pennylane as qml

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_DIR    = "models"
REPORTS_DIR  = "module9/reports"
TEMP_DIR     = "module9/temp"

WINDOW_SIZE  = 8
FRAME_SIZE   = 224
PATCH_GRID   = 4
NUM_QUBITS   = 3
LAMBDA       = 0.3

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,    exist_ok=True)

# ─────────────────────────────────────────────
# QUANTUM CIRCUIT (same as Module 5)
# ─────────────────────────────────────────────
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def quantum_circuit(feature_vector):
    for i in range(NUM_QUBITS):
        qml.RX(np.pi * feature_vector[i], wires=i)
        qml.RY(np.pi * feature_vector[i], wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# ---------------------------------------------
# STEP 1 - Extract frames from video
# ---------------------------------------------
def extract_frames(video_path):
    cap    = cv2.VideoCapture(video_path)
    frames = []
    fps    = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(frame)

    cap.release()
    return frames, fps


def create_windows(frames):
    windows = []
    for i in range(0, len(frames), WINDOW_SIZE):
        chunk = frames[i:i + WINDOW_SIZE]
        if len(chunk) == WINDOW_SIZE:
            windows.append(np.array(chunk))
    return windows


# ---------------------------------------------
# STEP 2 - Extract patch features (Module 2)
# ---------------------------------------------
def divide_into_patches(frame, grid):
    h, w     = frame.shape[:2]
    patch_h  = h // grid
    patch_w  = w // grid
    patches  = []
    for i in range(grid):
        for j in range(grid):
            patches.append(frame[i*patch_h:(i+1)*patch_h,
                                  j*patch_w:(j+1)*patch_w])
    return patches


def extract_patch_features(window):
    num_patches     = PATCH_GRID * PATCH_GRID
    window_features = np.zeros((WINDOW_SIZE, num_patches, 5), dtype=np.float32)

    for t in range(WINDOW_SIZE):
        patches = divide_into_patches(window[t], PATCH_GRID)
        for p, patch in enumerate(patches):
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            window_features[t, p, 0] = float(np.mean(gray))
            window_features[t, p, 1] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            window_features[t, p, 2] = float(shannon_entropy(gray))
            edges = cv2.Canny(gray, 100, 200)
            window_features[t, p, 3] = float(np.sum(edges > 0) / edges.size)

    for t in range(1, WINDOW_SIZE):
        prev_gray = cv2.cvtColor(window[t-1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(window[t],   cv2.COLOR_BGR2GRAY)
        flow      = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_patches = divide_into_patches(mag, PATCH_GRID)
        for p in range(num_patches):
            window_features[t, p, 4] = float(np.mean(mag_patches[p]))

    return window_features


# ---------------------------------------------
# STEP 3 - Quantum features (Module 5)
# ---------------------------------------------
def extract_quantum_features(patch_features):
    # Average over time axis → (P, F)
    pf         = patch_features.mean(axis=0)
    pf         = np.nan_to_num(pf)
    min_val    = pf.min(axis=0)
    max_val    = pf.max(axis=0)
    pf         = (pf - min_val) / (max_val - min_val + 1e-8)

    q_features = []
    for patch_feat in pf:
        q_feat = quantum_circuit(patch_feat[:NUM_QUBITS])
        q_features.append(q_feat)

    return np.array(q_features)


# ---------------------------------------------
# STEP 4 - TCD score (Module 4)
# ---------------------------------------------
def compute_tcd(curr_feats, prev_feats):
    if curr_feats.shape[0] < 2:
        return 0.0

    curr_feats = np.nan_to_num(curr_feats)
    prev_feats = np.nan_to_num(prev_feats)

    delta     = curr_feats - prev_feats
    mean_delta = np.mean(np.abs(np.linalg.norm(delta, axis=1)))

    distances = []
    for i, j in combinations(range(len(curr_feats)), 2):
        distances.append(np.linalg.norm(curr_feats[i] - curr_feats[j]))

    var_dist = np.var(distances) if distances else 0.0
    return var_dist + LAMBDA * mean_delta


# ---------------------------------------------
# STEP 5 - Build feature vector (Module 6)
# ---------------------------------------------
def build_feature_vector(patch_features, quantum_features, tcd_value):
    # patch_features shape: (T, P, F)
    if patch_features.ndim == 2:
        patch_features = patch_features[np.newaxis, ...]

    mean_emb   = quantum_features.mean(axis=0)
    std_emb    = quantum_features.std(axis=0)

    delta      = np.abs(patch_features[1:] - patch_features[:-1])
    mean_delta = float(delta.mean()) if patch_features.shape[0] > 1 else 0.0
    var_delta  = float(delta.var())  if patch_features.shape[0] > 1 else 0.0

    return np.concatenate([
        [tcd_value, mean_delta, var_delta],
        mean_emb,
        std_emb
    ])


# ---------------------------------------------
# MAIN INFERENCE
# ---------------------------------------------
def run(video_path):
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at '{video_path}'")
        return

    # ── Load model and scaler ────────────────────────────────────────────────
    model_path  = os.path.join(MODEL_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("ERROR: Model files not found. Run Module 8 first.")
        return

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"\nAnalysing video: {video_path}")
    print("=" * 50)

    # ── Extract frames and windows ───────────────────────────────────────────
    frames, fps = extract_frames(video_path)
    windows     = create_windows(frames)

    print(f"Total frames  : {len(frames)}")
    print(f"FPS           : {fps:.1f}")
    print(f"Total windows : {len(windows)}")
    print("Processing...")

    # ── Process each window ──────────────────────────────────────────────────
    window_results = []
    prev_patch_features = None

    for i, window in enumerate(windows):
        # Module 2
        patch_features = extract_patch_features(window)

        # Module 4 — TCD (needs previous window)
        if prev_patch_features is not None:
            pf_mean      = patch_features.mean(axis=0)
            prev_pf_mean = prev_patch_features.mean(axis=0)
            tcd_value    = compute_tcd(pf_mean, prev_pf_mean)
        else:
            tcd_value = 0.0

        # Module 5
        quantum_features = extract_quantum_features(patch_features)

        # Module 6
        feature_vector = build_feature_vector(
            patch_features, quantum_features, tcd_value)

        # Scale and predict
        feature_vector_scaled = scaler.transform([feature_vector])
        prediction            = model.predict(feature_vector_scaled)[0]
        probability           = model.predict_proba(feature_vector_scaled)[0]

        # Time range of this window
        start_sec = (i * WINDOW_SIZE) / fps
        end_sec   = ((i + 1) * WINDOW_SIZE) / fps

        window_results.append({
            "window"     : i,
            "start_sec"  : round(start_sec, 2),
            "end_sec"    : round(end_sec, 2),
            "prediction" : int(prediction),
            "label"      : "TAMPERED" if prediction == 1 else "NORMAL",
            "confidence" : round(float(max(probability)) * 100, 2),
            "tcd_score"  : round(tcd_value, 5),
        })

        prev_patch_features = patch_features

    # ── Overall verdict ──────────────────────────────────────────────────────
    total_windows    = len(window_results)
    tampered_windows = sum(1 for w in window_results if w["prediction"] == 1)
    tampered_ratio   = tampered_windows / total_windows if total_windows > 0 else 0

    # A video is flagged as tampered if >20% of windows are tampered
    overall_verdict  = "TAMPERED" if tampered_ratio > 0.20 else "NORMAL"
    avg_confidence   = np.mean([w["confidence"] for w in window_results])

    # ── Terminal output ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  VERDICT: {overall_verdict}")
    print(f"{'='*50}")
    print(f"  Tampered windows : {tampered_windows} / {total_windows}")
    print(f"  Tampered ratio   : {tampered_ratio*100:.1f}%")
    print(f"  Avg confidence   : {avg_confidence:.1f}%")

    if tampered_windows > 0:
        print(f"\n  Suspicious regions:")
        for w in window_results:
            if w["prediction"] == 1:
                print(f"    Window {w['window']:>3} | "
                      f"{w['start_sec']:>6.2f}s - {w['end_sec']:>6.2f}s | "
                      f"Confidence: {w['confidence']}% | "
                      f"TCD: {w['tcd_score']}")

    # ── Save report ──────────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    report_path = os.path.join(REPORTS_DIR, f"{video_name}_{timestamp}.csv")

    df = pd.DataFrame(window_results)
    df["overall_verdict"]  = overall_verdict
    df["tampered_ratio"]   = round(tampered_ratio * 100, 2)
    df["avg_confidence"]   = round(avg_confidence, 2)
    df.to_csv(report_path, index=False)

    print(f"\n[OK] Report saved: {report_path}")
    print(f"\nModule 9 complete.")

    # Output verdict for frontend
    print(overall_verdict.lower())

    return overall_verdict, window_results


# ---------------------------------------------
# ENTRY POINT
# ---------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python module9/inference.py <path_to_video>")
        print("Example: python module9/inference.py test_videos/suspect.mp4")
    else:
        run(sys.argv[1])