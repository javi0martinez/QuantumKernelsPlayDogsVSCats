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
