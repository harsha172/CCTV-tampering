import os
import numpy as np
import pennylane as qml

PATCH_FEATURE_FOLDER = "module2/features"
QUANTUM_FEATURE_FOLDER = "module5/quantum_features"

NUM_QUBITS = 5
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def quantum_circuit(feature_vector):
    for i in range(NUM_QUBITS):
        qml.RX(np.pi * feature_vector[i], wires=i)
        qml.RY(np.pi * feature_vector[i], wires=i)

    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


def normalize_features(features):
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    return (features - min_val) / (max_val - min_val + 1e-8)


def process_window(patch_features):
    """
    patch_features: (P, F)
    returns: (P, Q)
    """
    patch_features = np.nan_to_num(patch_features)
    patch_features = normalize_features(patch_features)

    quantum_features = []
    for patch_feat in patch_features:
        q_feat = quantum_circuit(patch_feat[:NUM_QUBITS])
        quantum_features.append(q_feat)

    return np.array(quantum_features)


def run_qeff():

    os.makedirs(QUANTUM_FEATURE_FOLDER, exist_ok=True)

    frame_folders = sorted([
        f for f in os.listdir(PATCH_FEATURE_FOLDER)
        if os.path.isdir(os.path.join(PATCH_FEATURE_FOLDER, f))
    ])

    for frame in frame_folders:
        frame_input_path = os.path.join(PATCH_FEATURE_FOLDER, frame)
        frame_output_path = os.path.join(QUANTUM_FEATURE_FOLDER, frame)

        os.makedirs(frame_output_path, exist_ok=True)

        window_files = sorted([
            f for f in os.listdir(frame_input_path)
            if f.endswith(".npy")
        ])

        for file in window_files:
            input_path = os.path.join(frame_input_path, file)
            patch_features = np.load(input_path)

            # Handle accidental (T, P, F) tensors
            if patch_features.ndim == 3:
                patch_features = patch_features.mean(axis=0)

            if patch_features.shape[0] == 0:
                print(f"Skipping {frame}/{file} (empty patches)")
                continue

            quantum_features = process_window(patch_features)

            output_path = os.path.join(frame_output_path, file)
            np.save(output_path, quantum_features)

            print(f"Processed {frame}/{file} → {quantum_features.shape}")


if __name__ == "__main__":
    run_qeff()
