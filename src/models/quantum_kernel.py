"""
Quantum kernel implementation using PennyLane.
"""

import pennylane as qml
from pennylane import numpy as np


def create_quantum_kernel(num_wires=5, num_layers=3):
    """Create a quantum kernel for classification."""

    def layer(x, params, wires, i0=0, inc=1):
        """Build a layer of the quantum circuit for data embedding."""
        i = i0
        for j, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            qml.RY(params[0, j], wires=[wire])
            i += inc

        for j in range(len(wires)):
            qml.CRZ(params[1, j], wires=[wires[j], wires[(j + 1) % len(wires)]])

    def ansatz(x, params, wires):
        """Build the quantum circuit."""
        for j, layer_params in enumerate(params):
            layer(x, layer_params, wires, i0=j * len(wires))

    adjoint_ansatz = qml.adjoint(ansatz)

    def random_params(num_wires, num_layers):
        """Generate random variational parameters with the shape of ansatz."""
        return np.random.uniform(
            0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True
        )

    # Define the quantum device
    dev = qml.device("default.qubit", wires=num_wires, shots=None)
    wires = dev.wires.tolist()

    @qml.qnode(dev)
    def kernel_circuit(x1, x2, params):
        """Quantum circuit for kernel computation."""
        ansatz(x1, params, wires=wires)
        adjoint_ansatz(x2, params, wires=wires)
        return qml.probs(wires=wires)

    def kernel(x1, x2, params):
        """Compute the kernel value between two feature vectors."""
        return kernel_circuit(x1, x2, params)[0]

    return kernel, random_params
