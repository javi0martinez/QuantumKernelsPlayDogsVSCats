"""
Training utilities for both classical and quantum models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pennylane import numpy as np
from sklearn.svm import SVC
import pennylane as qml


def train_cnn(model, train_loader, val_loader, device, learning_rate=0.0001, epochs=5):
    """Train the CNN model."""

    model.train()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in train_loader:
            # Move to GPU if available
            data = data.to(device)
            label = label.to(device)

            # Forward pass
            output = model(data)[0]
            loss = criterion(output, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            accuracy = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += accuracy / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(
            f"Epoch : {epoch+1}, Training Accuracy : {epoch_accuracy}, Training Loss : {epoch_loss}"
        )

        # Validation
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)[0]
                val_loss = criterion(val_output, label)

                accuracy = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += accuracy / len(val_loader)
                epoch_val_loss += val_loss / len(val_loader)

            print(
                f"Epoch : {epoch+1}, Validation Accuracy : {epoch_val_accuracy}, Validation Loss : {epoch_val_loss}"
            )

    return model


def train_quantum_kernel(
    kernel,
    X_train,
    Y_train,
    X_test,
    Y_test,
    init_params=None,
    num_iterations=700,
    learning_rate=0.2,
    batch_size=8,
):
    """Train the quantum kernel."""

    def target_alignment(
        X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=True
    ):
        """Calculate the kernel target alignment."""
        K = qml.kernels.square_kernel_matrix(
            X, kernel, assume_normalized_kernel=assume_normalized_kernel
        )

        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(Y) == 1)
            nminus = len(Y) - nplus
            _Y = np.array([y / nplus if y == 1 else -1 / nminus for y in Y])
        else:
            _Y = np.array(Y)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        return inner_product / norm

    opt = qml.GradientDescentOptimizer(learning_rate)

    # Initialize parameters
    if init_params is None:
        raise ValueError("init_params must be provided")
    params = init_params

    # Convert to numpy arrays if needed
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Optimization loop
    for i in range(num_iterations):
        # Select subset for batch training
        subset = np.random.choice(list(range(len(X_train))), batch_size)

        # Cost function (negative KTA)
        cost = lambda _params: -target_alignment(
            X_train[subset],
            Y_train[subset],
            lambda x1, x2: kernel(x1, x2, _params),
            assume_normalized_kernel=True,
        )

        # Optimization step
        params = opt.step(cost, params)

        # Print progress
        if (i + 1) % 50 == 0:
            current_alignment = target_alignment(
                X_train,
                Y_train,
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
            print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

    # Train SVM with optimized kernel
    trained_kernel = lambda x1, x2: kernel(x1, x2, params)
    trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(
        X1, X2, trained_kernel
    )
    svm = SVC(kernel=trained_kernel_matrix).fit(X_train, Y_train)

    return svm, trained_kernel
