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

        # Print progress - calculate alignment on a subset to avoid computational bottleneck
        if (i + 1) % 50 == 0:
            # Use a larger subset for alignment evaluation (but not the full dataset)
            eval_size = min(50, len(X_train))
            eval_subset = np.random.choice(list(range(len(X_train))), eval_size, replace=False)
            current_alignment = target_alignment(
                X_train[eval_subset],
                Y_train[eval_subset],
                lambda x1, x2: kernel(x1, x2, params),
                assume_normalized_kernel=True,
            )
            print(f"Step {i+1}/{num_iterations} - Alignment (subset) = {current_alignment:.3f}")

    # Train SVM with optimized kernel
    print("\nOptimization complete. Training SVM...")
    print(f"Dataset size: {len(X_train)} samples")
    
    trained_kernel = lambda x1, x2: kernel(x1, x2, params)
    
    # For large datasets, use a subset for SVM training to avoid computational bottleneck
    max_svm_samples = 200  # Adjust based on computational resources
    if len(X_train) > max_svm_samples:
        print(f"Using subset of {max_svm_samples} samples for SVM training...")
        svm_indices = np.random.choice(len(X_train), max_svm_samples, replace=False)
        X_train_svm = X_train[svm_indices]
        Y_train_svm = Y_train[svm_indices]
    else:
        X_train_svm = X_train
        Y_train_svm = Y_train
    
    # Pre-compute kernel matrix for training
    print("Computing kernel matrix...")
    K_train = qml.kernels.square_kernel_matrix(X_train_svm, trained_kernel)
    
    # Train SVM with precomputed kernel
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, Y_train_svm)
    
    print("SVM training complete!")

    # Return SVM, params (not lambda), and training data (needed for precomputed kernel predictions)
    return svm, params, X_train_svm


def predict_with_quantum_kernel(svm, kernel, X_train_svm, X_test):
    """
    Make predictions using SVM trained with precomputed kernel.
    
    Args:
        svm: Trained SVM model (with kernel='precomputed')
        kernel: The trained quantum kernel function
        X_train_svm: Training data used to fit the SVM
        X_test: Test data to predict
        
    Returns:
        predictions: Array of predicted labels
    """
    print(f"Computing kernel matrix for {len(X_test)} test samples...")
    # Compute kernel matrix between test and training data
    K_test = qml.kernels.kernel_matrix(X_test, X_train_svm, kernel)
    predictions = svm.predict(K_test)
    return predictions
