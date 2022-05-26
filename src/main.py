"""
Main script to run the hybrid quantum-classical image classifier.
"""

import os
import torch
from data.data_loading import prepare_data_loaders
from models.cnn_model import CnnFeatureExtractor
from models.quantum_kernel import create_quantum_kernel
from utils.training import train_cnn, train_quantum_kernel


def main():
    """Main function to run the training and evaluation pipeline."""

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(333)
    if device == "cuda":
        torch.cuda.manual_seed_all(333)

    # Hyperparameters
    batch_size = 100
    learning_rate = 0.0001
    epochs = 5
    img_size = 224

    # Prepare data
    data_dir = os.path.join("data", "DogsAndCats")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir, batch_size, img_size
    )

    # Train CNN
    print("Training CNN...")
    model = CnnFeatureExtractor().to(device)
    model = train_cnn(model, train_loader, val_loader, device, learning_rate, epochs)

    # Extract features using CNN
    print("Extracting features...")
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    with torch.no_grad():
        for data, label in train_loader:
            _, features = model(data.to(device))
            train_features.extend(features.cpu().numpy())
            train_labels.extend(label.numpy())

        for data, label in val_loader:
            _, features = model(data.to(device))
            val_features.extend(features.cpu().numpy())
            val_labels.extend(label.numpy())

    # Train quantum kernel
    kernel, random_params = create_quantum_kernel()
    init_params = random_params(num_wires=5, num_layers=3)

    svm, trained_kernel = train_quantum_kernel(
        kernel,
        train_features[:400],
        train_labels[:400],
        val_features[400:1400],
        val_labels[400:1400],
        init_params=init_params,
    )

