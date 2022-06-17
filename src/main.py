"""
Main script to run the hybrid quantum-classical image classifier.
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import torch
from data.data_loading import prepare_data_loaders
from models.cnn_model import CnnFeatureExtractor
from models.quantum_kernel import create_quantum_kernel
from utils.training import train_cnn, train_quantum_kernel, predict_with_quantum_kernel


def main(retrain_cnn=False, retrain_quantum=False):
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

    # Initialize model
    model = CnnFeatureExtractor().to(device)
    cnn_model_path = os.path.join("models", "classical", "cnn_model.pth")

    # Load or train CNN
    if not retrain_cnn and os.path.exists(cnn_model_path):
        print(f"Loading pre-trained CNN from {cnn_model_path}...")
        model.load_state_dict(torch.load(cnn_model_path))
        model.eval()
    else:
        print("Training CNN...")
        model = train_cnn(model, train_loader, val_loader, device, learning_rate, epochs)
        # Save the trained model
        print(f"Saving CNN model to {cnn_model_path}...")
        os.makedirs(os.path.dirname(cnn_model_path), exist_ok=True)
        torch.save(model.state_dict(), cnn_model_path)

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

    # Initialize or load quantum kernel
    quantum_model_path = os.path.join("models", "quantum", "quantum_kernel.pth")
    
    # Create kernel function (needed for both loading and training)
    kernel, random_params = create_quantum_kernel()
    
    # Load or train quantum kernel
    if not retrain_quantum and os.path.exists(quantum_model_path):
        print(f"Loading pre-trained quantum kernel from {quantum_model_path}...")
        quantum_data = torch.load(quantum_model_path)
        kernel_params = quantum_data["kernel_params"]
        svm = quantum_data["svm_state"]
        X_train_svm = quantum_data["X_train_svm"]
    else:
        print("Training quantum kernel...")
        init_params = random_params(num_wires=5, num_layers=3)

        svm, kernel_params, X_train_svm = train_quantum_kernel(
            kernel,
            train_features[:400],
            train_labels[:400],
            val_features[400:1400],
            val_labels[400:1400],
            init_params=init_params,
        )
        
        # Save the trained quantum kernel
        print(f"\nSaving quantum kernel to {quantum_model_path}...")
        os.makedirs(os.path.dirname(quantum_model_path), exist_ok=True)
        torch.save(
            {
                "kernel_params": kernel_params, 
                "svm_state": svm,
                "X_train_svm": X_train_svm
            },
            quantum_model_path,
        )

    # Reconstruct trained kernel function from parameters
    trained_kernel = lambda x1, x2: kernel(x1, x2, kernel_params)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = predict_with_quantum_kernel(
        svm, trained_kernel, X_train_svm, val_features[:100]
    )
    val_accuracy = (val_predictions == val_labels[:100]).mean()
    print(f"Validation Accuracy: {val_accuracy:.3f}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid quantum-classical image classifier")
    parser.add_argument(
        "--retrain-cnn",
        action="store_true",
        help="Retrain the CNN model even if a saved model exists",
    )
    parser.add_argument(
        "--retrain-quantum",
        action="store_true",
        help="Retrain the quantum kernel even if a saved model exists",
    )
    parser.add_argument(
        "--retrain-all",
        action="store_true",
        help="Retrain both CNN and quantum kernel models",
    )
    args = parser.parse_args()
    
    # If --retrain-all is set, retrain both models
    retrain_cnn = args.retrain_cnn or args.retrain_all
    retrain_quantum = args.retrain_quantum or args.retrain_all
    
    main(retrain_cnn=retrain_cnn, retrain_quantum=retrain_quantum)
