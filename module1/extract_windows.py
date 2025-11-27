import cv2
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_FOLDER = "module1/input_videos"
OUTPUT_FOLDER = "module1/windows"
WINDOW_SIZE = 8   # Number of consecutive frames per window
FRAME_WIDTH = 224
FRAME_HEIGHT = 224


def extract_frames(video_path):
    """Loads video and extracts resized frames."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(frame)

    cap.release()
    return frames


def create_windows(frames, window_size):
    """Creates windows of N consecutive frames."""
    windows = []
    for i in range(0, len(frames), window_size):
        chunk = frames[i:i + window_size]
        if len(chunk) == window_size:
            windows.append(np.array(chunk))
    return windows


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith((".mp4", ".avi"))]

    if not video_files:
        print("❌ No videos found in input_videos/")
        return

    for video in video_files:
        print(f"▶ Processing: {video}")

        video_path = os.path.join(VIDEO_FOLDER, video)

        # 1. Extract frames
        frames = extract_frames(video_path)
        print(f"   ✔ Extracted {len(frames)} frames")

        # 2. Create windows
        windows = create_windows(frames, WINDOW_SIZE)
        print(f"   ✔ Created {len(windows)} windows")

        # 3. Save each window
        base_name = video.split(".")[0]
        for i, win in enumerate(windows):
            out_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_window{i}.npy")
            np.save(out_path, win)

        print("   ✔ Saved windows successfully!\n")


if __name__ == "__main__":
    main()
