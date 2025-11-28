import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

GRAPH_FOLDER = "module3/graphs"

# Get all adjacency matrix files
files = sorted([f for f in os.listdir(GRAPH_FOLDER) if f.endswith(".npy")])

if not files:
    print("❌ No graph files found in", GRAPH_FOLDER)
    exit()

for file in files:
    path = os.path.join(GRAPH_FOLDER, file)
    adj = np.load(path)  # load adjacency matrix

    # Convert to NetworkX graph
    G = nx.from_numpy_array(adj)

    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)  # position nodes nicely
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')

    # Draw edge weights
    edge_labels = nx.get_edge_attributes(G, 'weight')
    rounded_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=rounded_labels, font_size=8)

    plt.title(file)
    plt.show()
