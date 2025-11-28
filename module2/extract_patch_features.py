import os
import cv2
import numpy as np
from skimage.measure import shannon_entropy

# ---------------------------
# CONFIGURATION
# ---------------------------
WINDOW_FOLDER = "module1/windows"
OUTPUT_FOLDER = "module2/features"
PATCH_GRID = 4           # 4×4 = 16 patches
FRAME_SIZE = 224
WINDOW_SIZE = 8          # must match module1


def divide_into_patches(frame, grid):
    """Splits a frame into N×N patches."""
    h, w = frame.shape[:2]
    patch_h = h // grid
    patch_w = w // grid

    patches = []
    for i in range(grid):
        for j in range(grid):
            patch = frame[
                i * patch_h:(i + 1) * patch_h,
                j * patch_w:(j + 1) * patch_w
            ]
            patches.append(patch)
    return patches


def compute_brightness(patch):
    return float(np.mean(patch))


def compute_blur(patch):
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def compute_entropy(patch):
    return float(shannon_entropy(patch))


def compute_edge_density(patch):
    edges = cv2.Canny(patch, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def compute_optical_flow(prev_frame, curr_frame):
    """Returns magnitude of optical flow between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag


def extract_features_from_window(window):
    """
    Returns numeric 3D array:
    shape = (WINDOW_SIZE, 16 patches, 5 features)
    """
    num_patches = PATCH_GRID * PATCH_GRID

    # initialize full array
    # window_features[t][p] = [brightness, blur, entropy, edge, flow]
    window_features = np.zeros((WINDOW_SIZE, num_patches, 5), dtype=np.float32)

    # --- Step 1: Compute all features except optical flow ---
    for t in range(WINDOW_SIZE):
        frame = window[t]
        patches = divide_into_patches(frame, PATCH_GRID)

        for p, patch in enumerate(patches):
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            window_features[t, p, 0] = compute_brightness(gray)
            window_features[t, p, 1] = compute_blur(gray)
            window_features[t, p, 2] = compute_entropy(gray)
            window_features[t, p, 3] = compute_edge_density(gray)
            # flow added later

    # --- Step 2: Compute optical flow for frames 1..7 ---
    for t in range(1, WINDOW_SIZE):
        prev_frame = window[t - 1]
        curr_frame = window[t]

        mag = compute_optical_flow(prev_frame, curr_frame)
        mag_patches = divide_into_patches(mag, PATCH_GRID)

        for p in range(num_patches):
            window_features[t, p, 4] = float(np.mean(mag_patches[p]))

    return window_features


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    window_files = [f for f in os.listdir(WINDOW_FOLDER) if f.endswith(".npy")]

    for window_file in window_files:
        print(f"▶ Processing window: {window_file}")

        window_path = os.path.join(WINDOW_FOLDER, window_file)
        window = np.load(window_path)  # shape: (8, 224, 224, 3)

        features = extract_features_from_window(window)
        # shape = (8, 16, 5)

        out_file = window_file.replace(".npy", "_features.npy")
        out_path = os.path.join(OUTPUT_FOLDER, out_file)

        np.save(out_path, features)

        print("   ✔ Saved numeric feature file:", out_file)


if __name__ == "__main__":
    main()
