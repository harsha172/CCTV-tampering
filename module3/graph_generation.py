import os
import numpy as np
import networkx as nx

# --------------------------
# CONFIG
# --------------------------
PATCH_FEATURE_FOLDER = "module2/features"
OUTPUT_FOLDER = "module3/graphs"
PATCH_GRID = 4  # Needed to know neighbors → 4x4 = 16 patches


# --------------------------
# HELPER: Get neighboring patches
# --------------------------
def get_neighbors(patch_id, grid):
    neighbors = []
    row = patch_id // grid
    col = patch_id % grid

    # Up
    if row > 0:
        neighbors.append((row - 1) * grid + col)

    # Down
    if row < grid - 1:
        neighbors.append((row + 1) * grid + col)

    # Left
    if col > 0:
        neighbors.append(row * grid + (col - 1))

    # Right
    if col < grid - 1:
        neighbors.append(row * grid + (col + 1))

    return neighbors


# --------------------------
# MODULE 3: Build Graphs
# --------------------------
def build_dynamic_graph(patch_features):
    """
    patch_features shape = (num_windows, num_patches, num_features)
    Example: (12 windows, 16 patches, 5 features)
    """

    graphs = []

    for w in range(patch_features.shape[0]):
        G = nx.Graph()

        # ADD NODES
        for p in range(patch_features.shape[1]):
            G.add_node(p, features=patch_features[w, p])

        # ADD EDGES
        for p in range(patch_features.shape[1]):
            neighbors = get_neighbors(p, PATCH_GRID)

            for nb in neighbors:
                # Compute correlation as edge weight
                corr = np.corrcoef(
                    patch_features[w, p], patch_features[w, nb]
                )[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                G.add_edge(p, nb, weight=float(corr))

        graphs.append(G)

    return graphs


# --------------------------
# MAIN FUNCTION
# --------------------------


# --------------------------
# MAIN FUNCTION
# --------------------------
def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Loop through each video subfolder in module2/features
    for video_name in os.listdir(PATCH_FEATURE_FOLDER):
        video_folder = os.path.join(PATCH_FEATURE_FOLDER, video_name)
        if not os.path.isdir(video_folder):
            continue

        files = sorted([f for f in os.listdir(video_folder) if f.endswith(".npy")])
        if not files:
            print(f"No patch feature files found for {video_name}")
            continue

        print(f"▶ Processing video: {video_name}")

        # Create matching output folder for graphs
        out_video_folder = os.path.join(OUTPUT_FOLDER, video_name)
        os.makedirs(out_video_folder, exist_ok=True)

        for file in files:
            print(f"Processing: {file}")
            path = os.path.join(video_folder, file)
            patch_features = np.load(path, allow_pickle=True)   # shape = (num_windows, 16, 5)

            graphs = build_dynamic_graph(patch_features)

            base = file.replace(".npy", "")
            for i, G in enumerate(graphs):
                A = nx.to_numpy_array(G)
                out_path = os.path.join(out_video_folder, f"{base}_window{i}_adj.npy")
                np.save(out_path, A)

            print(f"Saved {len(graphs)} graphs for {file} into {out_video_folder}")

if __name__ == "__main__":
    main()
