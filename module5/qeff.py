import os
import numpy as np
import pennylane as qml

PATCH_FEATURE_FOLDER = "module2/features"
QUANTUM_FEATURE_FOLDER = "module5/quantum_features"

NUM_QUBITS = 3
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev)
def quantum_circuit(feature_vector):
    """
    feature_vector: shape (F,)
    returns: quantum embedding (Q,)
    """

    # -------- Angle Encoding --------
    for i in range(NUM_QUBITS):
        qml.RX(np.pi * feature_vector[i], wires=i)
        qml.RY(np.pi * feature_vector[i], wires=i)

    # -------- Entanglement --------
    for i in range(NUM_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # -------- Measurement --------
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]

def normalize_features(features):
    """
    Normalize features to [0, 1] range
    """
    min_val = features.min(axis=0)
    max_val = features.max(axis=0)
    return (features - min_val) / (max_val - min_val + 1e-8)

def process_window(patch_features):
    """
    patch_features: shape (P, F)
    returns: quantum_features shape (P, Q)
    """
    patch_features = normalize_features(patch_features)

    quantum_features = []

    for patch_feat in patch_features:
        q_feat = quantum_circuit(patch_feat[:NUM_QUBITS])
        quantum_features.append(q_feat)

    return np.array(quantum_features)

def run_qeff():
    if not os.path.exists(QUANTUM_FEATURE_FOLDER):
        os.makedirs(QUANTUM_FEATURE_FOLDER)

    files = sorted([f for f in os.listdir(PATCH_FEATURE_FOLDER) if f.endswith(".npy")])

    for file in files:
        input_path = os.path.join(PATCH_FEATURE_FOLDER, file)
        patch_features = np.load(input_path)

        # Aggregate over frames if needed
        if patch_features.ndim == 3:
            patch_features = patch_features.mean(axis=0)  # (P, F)

        quantum_features = process_window(patch_features)

        output_path = os.path.join(QUANTUM_FEATURE_FOLDER, file)
        np.save(output_path, quantum_features)

        print(f"Processed {file} → {quantum_features.shape}")
 
if __name__ == "__main__":
    run_qeff()
    q = np.load("module5/quantum_features/tampered_window8_features.npy")
    print(q)

