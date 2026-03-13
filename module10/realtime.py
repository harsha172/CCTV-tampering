import cv2
import numpy as np
import joblib
import threading
import queue
import time
from collections import deque
from itertools import combinations
from skimage.measure import shannon_entropy
import pennylane as qml
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← tweak these if verdict is still wrong
# ─────────────────────────────────────────────────────────────
MODEL_DIR    = "models"
WINDOW_SIZE  = 8
FRAME_SIZE   = 112
PATCH_GRID   = 4
NUM_QUBITS   = 3
LAMBDA       = 0.3
CAMERA_INDEX = 0

# ── Verdict sensitivity controls ─────────────────────────────
# How many of the last N windows must be TAMPERED to flip verdict
SMOOTH_TOTAL    = 5    # look at last N windows
SMOOTH_REQUIRED = 5    # need this many tampered out of N to say TAMPERED

# Once TAMPERED, how many consecutive NORMAL windows to flip back
RESET_REQUIRED  = 4

# Tampered predictions below this confidence % are ignored (treated as normal)
MIN_TAMPERED_CONFIDENCE = 90.0

# ─────────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_DUPLEX


# ─────────────────────────────────────────────────────────────
# QUANTUM CIRCUIT
# ─────────────────────────────────────────────────────────────
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def quantum_circuit(fv):
    for i in range(NUM_QUBITS):
        qml.RX(np.pi * float(fv[i]), wires=i)
        qml.RY(np.pi * float(fv[i]), wires=i)
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def divide_patches(frame, grid):
    h, w = frame.shape[:2]
    ph, pw = h // grid, w // grid
    return [frame[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            for i in range(grid) for j in range(grid)]


def patch_features(window):
    T, P = WINDOW_SIZE, PATCH_GRID * PATCH_GRID
    feats = np.zeros((T, P, 5), dtype=np.float32)
    for t in range(T):
        patches = divide_patches(window[t], PATCH_GRID)
        for p, patch in enumerate(patches):
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            feats[t, p, 0] = np.mean(gray)
            feats[t, p, 1] = cv2.Laplacian(gray, cv2.CV_64F).var()
            feats[t, p, 2] = float(shannon_entropy(gray))
            edges = cv2.Canny(gray, 100, 200)
            feats[t, p, 3] = np.sum(edges > 0) / edges.size
    for t in range(1, T):
        pg = cv2.cvtColor(window[t-1], cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(window[t],   cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        for p, mp in enumerate(divide_patches(mag, PATCH_GRID)):
            feats[t, p, 4] = np.mean(mp)
    return feats


def quantum_embed(pf_mean):
    pf = np.nan_to_num(pf_mean)
    mn, mx = pf.min(0), pf.max(0)
    pf = (pf - mn) / (mx - mn + 1e-8)
    return np.array([quantum_circuit(row[:NUM_QUBITS]) for row in pf])


def tcd_score(curr, prev):
    curr, prev = np.nan_to_num(curr), np.nan_to_num(prev)
    delta  = curr - prev
    mean_d = np.mean(np.abs(np.linalg.norm(delta, axis=1)))
    dists  = [np.linalg.norm(curr[i] - curr[j])
              for i, j in combinations(range(len(curr)), 2)]
    return float(np.var(dists) + LAMBDA * mean_d) if dists else 0.0


def build_vector(pf, qf, tcd):
    mean_emb = qf.mean(axis=0)
    std_emb  = qf.std(axis=0)
    if pf.shape[0] > 1:
        delta = np.abs(pf[1:] - pf[:-1])
        md, vd = float(delta.mean()), float(delta.var())
    else:
        md, vd = 0.0, 0.0
    return np.concatenate([[tcd, md, vd], mean_emb, std_emb])


def process_window(window, prev_pf_mean):
    pf      = patch_features(window)
    pf_mean = pf.mean(axis=0)
    tcd     = tcd_score(pf_mean, prev_pf_mean) if prev_pf_mean is not None else 0.0
    qf      = quantum_embed(pf_mean)
    vec     = build_vector(pf, qf, tcd)
    return pf_mean, vec


# ─────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame, verdict, confidence, fps, processing_ms, history, normal_streak):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if verdict == "ANALYSING...":
        color = (180, 180, 0)
    elif verdict == "TAMPERED":
        color = (0, 0, 255)
        if int(time.time() * 3) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
    else:
        color = (0, 210, 0)

    cv2.putText(frame, verdict, (20, 62), FONT, 1.8, color, 3, cv2.LINE_AA)

    if confidence is not None:
        cv2.putText(frame, f"{confidence:.0f}% conf", (w-220, 48),
                    FONT, 0.75, (210, 210, 210), 1, cv2.LINE_AA)

    # Reset progress bar
    if verdict == "TAMPERED" and normal_streak > 0:
        bar_w = int((normal_streak / RESET_REQUIRED) * 200)
        cv2.rectangle(frame, (w-220, 60), (w-20, 80), (60, 60, 60), -1)
        cv2.rectangle(frame, (w-220, 60), (w-220+bar_w, 80), (0, 180, 0), -1)
        cv2.putText(frame, "Resetting...", (w-220, 95),
                    FONT, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Camera: {fps:.0f} fps", (15, h-45),
                FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Process: {processing_ms:.0f} ms", (15, h-20),
                FONT, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

    sq, gap  = 16, 3
    n_show   = 20
    start_x  = w - n_show*(sq+gap) - 10
    cv2.putText(frame, "History:", (start_x-70, h-28),
                FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    for i, v in enumerate(list(history)[-n_show:]):
        c = (0, 0, 200) if v == 1 else (0, 180, 0) if v == 0 else (80, 80, 80)
        x = start_x + i*(sq+gap)
        cv2.rectangle(frame, (x, h-44), (x+sq, h-12), c, -1)

    info = f"Sens: {SMOOTH_REQUIRED}/{SMOOTH_TOTAL} | MinConf: {MIN_TAMPERED_CONFIDENCE:.0f}%"
    cv2.putText(frame, info, (w//2-160, h-10), FONT, 0.42, (100,100,100), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q: quit", (w-80, h-10), FONT, 0.42, (100,100,100), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────
# BACKGROUND PROCESSOR
# ─────────────────────────────────────────────────────────────
class Processor:
    def __init__(self, model, scaler):
        self.model   = model
        self.scaler  = scaler
        self.in_q    = queue.Queue(maxsize=2)
        self.out_q   = queue.Queue(maxsize=10)
        self.prev_pf = None
        self.thread  = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, window):
        try:
            self.in_q.put_nowait(window)
        except queue.Full:
            pass

    def _run(self):
        while True:
            window = self.in_q.get()
            t0 = time.time()
            try:
                pf_mean, vec  = process_window(window, self.prev_pf)
                self.prev_pf  = pf_mean
                vec_scaled    = self.scaler.transform([vec])
                pred          = int(self.model.predict(vec_scaled)[0])
                proba         = self.model.predict_proba(vec_scaled)[0]
                conf          = float(max(proba)) * 100
                ms            = (time.time() - t0) * 1000
                self.out_q.put({"pred": pred, "conf": conf, "ms": ms})
            except Exception as e:
                print(f"Processing error: {e}")
                self.out_q.put({"pred": -1, "conf": None, "ms": 0})

    def latest_result(self):
        result = None
        while not self.out_q.empty():
            result = self.out_q.get_nowait()
        return result


# ─────────────────────────────────────────────────────────────
# VERDICT STATE MACHINE
# ─────────────────────────────────────────────────────────────
class VerdictState:
    def __init__(self):
        self.verdict       = "ANALYSING..."
        self.smooth_buf    = deque(maxlen=SMOOTH_TOTAL)
        self.normal_streak = 0

    def update(self, pred, conf):
        # Ignore low-confidence tampered predictions
        if pred == 1 and (conf is None or conf < MIN_TAMPERED_CONFIDENCE):
            pred = 0

        self.smooth_buf.append(pred)
        tampered_count = sum(self.smooth_buf)

        if self.verdict == "ANALYSING...":
            if len(self.smooth_buf) >= SMOOTH_TOTAL:
                self.verdict = "TAMPERED" if tampered_count >= SMOOTH_REQUIRED else "NORMAL"

        elif self.verdict == "NORMAL":
            if len(self.smooth_buf) == SMOOTH_TOTAL and tampered_count >= SMOOTH_REQUIRED:
                self.verdict       = "TAMPERED"
                self.normal_streak = 0

        elif self.verdict == "TAMPERED":
            if pred == 0:
                self.normal_streak += 1
            else:
                self.normal_streak = 0

            if self.normal_streak >= RESET_REQUIRED:
                self.verdict       = "NORMAL"
                self.normal_streak = 0
                self.smooth_buf.clear()

        return self.verdict, self.normal_streak


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run():
    import os

    if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
        print("ERROR: Model not found. Run Module 8 first.")
        return

    print("Loading model...")
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("Model loaded. Opening camera...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"Ready. Sensitivity: {SMOOTH_REQUIRED}/{SMOOTH_TOTAL} windows | "
          f"Min confidence: {MIN_TAMPERED_CONFIDENCE}% | "
          f"Reset after: {RESET_REQUIRED} normal windows")
    print("Press Q to quit.")

    processor  = Processor(model, scaler)
    buf        = deque(maxlen=WINDOW_SIZE)
    history    = deque(maxlen=20)
    state      = VerdictState()

    display_confidence = None
    display_ms         = 0.0
    normal_streak      = 0
    fps_counter        = 0
    fps_timer          = time.time()
    display_fps        = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            display_fps = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()

        buf.append(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)))

        if len(buf) == WINDOW_SIZE:
            processor.submit(np.array(buf))

        result = processor.latest_result()
        if result is not None and result["pred"] in (0, 1):
            display_confidence = result["conf"]
            display_ms         = result["ms"]
            verdict, normal_streak = state.update(result["pred"], result["conf"])
            history.append(result["pred"])
        else:
            verdict = state.verdict

        frame = draw_overlay(frame, verdict, display_confidence,
                             display_fps, display_ms, history, normal_streak)
        cv2.imshow("Video Tampering Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    run()