# Hybrid Quantum-Classical Image Classifier

A hybrid machine learning approach combining classical Convolutional Neural Networks (CNNs) with Quantum Kernels for binary image classification on the Dogs vs Cats dataset.

## 🎯 Overview

This project demonstrates the integration of classical deep learning with quantum computing for image classification. The approach leverages:

- **Classical CNN**: For feature extraction from raw images
- **Quantum Kernel**: For classification using quantum circuits via PennyLane
- **Support Vector Machine (SVM)**: For final classification with the quantum kernel

The hybrid architecture exploits the strengths of both paradigms:
- CNNs excel at extracting meaningful features from complex visual data
- Quantum kernels can capture intricate relationships in high-dimensional feature spaces

## 📁 Project Structure

```
QuantumKernelsPlayDogsVSCats/
├── data/
│   └── DogsAndCats/          # Dataset directory
│       ├── train/            # Training images
│       └── test/             # Test images
├── models/
│   ├── classical/            # Saved CNN models
│   │   └── cnn_model.pth
│   └── quantum/              # Saved quantum kernel models
│       └── quantum_kernel.pth
├── notebooks/
│   ├── HybridQuantumClassifier_Visual.ipynb  # Interactive visual notebook
│   └── old/                  # Legacy notebooks
├── src/
│   ├── main.py               # Main training script
│   ├── api/
│   │   └── kaggle.py         # Kaggle API utilities
│   ├── data/
│   │   └── data_loading.py   # Data loading and preprocessing
│   ├── models/
│   │   ├── cnn_model.py      # CNN architecture
│   │   └── quantum_kernel.py # Quantum kernel implementation
│   └── utils/
│       └── training.py       # Training utilities
├── environment.yaml          # Conda environment specification
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Anaconda or Miniconda
- Python 3.10
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd QuantumKernelsPlayDogsVSCats
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yaml
   conda activate QuantumKernelsPlayDogsVSCats
   ```

3. **Install additional dependencies**
   ```bash
   pip install seaborn tqdm
   ```

4. **Download the dataset**
   
   The project uses the Dogs vs Cats dataset from Kaggle. Place the images in the following structure:
   ```
   data/DogsAndCats/
   ├── train/
   │   ├── cat.0.jpg
   │   ├── dog.0.jpg
   │   └── ...
   └── test/
       ├── cat.100.jpg
       ├── dog.100.jpg
       └── ...
   ```

## 💻 Usage

### Option 1: Run the Main Script

Run the complete training pipeline from the command line:

```bash
# Run with pre-trained models (if available)
python src/main.py

# Retrain only the CNN
python src/main.py --retrain-cnn

# Retrain only the quantum kernel
python src/main.py --retrain-quantum

# Retrain both models
python src/main.py --retrain-all
```

### Option 2: Interactive Jupyter Notebook (Recommended)

For a more visual and interactive experience:

```bash
jupyter notebook notebooks/HybridQuantumClassifier_Visual.ipynb
```

The notebook provides:
- ✨ Visual data exploration with sample images
- 📊 Feature distribution analysis
- 🎨 Real-time training progress visualization
- 📈 Performance comparison charts
- 🔍 Confusion matrices and classification reports
- 🖼️ Prediction visualization with color-coded results
- ⚙️ Easy configuration controls (retrain flags, hyperparameters)

## 🏗️ Architecture

### 1. CNN Feature Extractor

A custom CNN architecture with:
- 3 convolutional layers (16, 32, 64 filters)
- Batch normalization and ReLU activation
- Max pooling for spatial dimension reduction
- Fully connected layers reducing to 10-dimensional features

**Output**: 10-dimensional feature vectors for each image

### 2. Quantum Kernel

Implemented using PennyLane:
- **Device**: `default.qubit`
- **Qubits**: 5 (configurable)
- **Layers**: 3 (configurable)
- **Gates**: Hadamard, RZ, RY, and CRZ (controlled rotation)
- **Training**: Kernel Target Alignment (KTA) optimization

The quantum circuit embeds classical features into quantum states and computes kernel values through quantum measurements.

### 3. SVM Classifier

Support Vector Machine with precomputed quantum kernel for final classification.

## 📊 Hyperparameters

### CNN Training
- **Batch Size**: 100
- **Learning Rate**: 0.0001
- **Epochs**: 5
- **Image Size**: 224x224

### Quantum Kernel Training
- **Number of Qubits**: 5
- **Number of Layers**: 3
- **Iterations**: 700
- **Learning Rate**: 0.2
- **Batch Size**: 8
- **Training Samples**: 400 (for computational efficiency)

## 🎯 Results

The hybrid quantum-classical model achieves competitive performance compared to the classical CNN baseline. Specific results depend on:
- Dataset size and quality
- Training iterations
- Quantum circuit architecture
- Hardware acceleration availability

Results are visualized in the interactive notebook with:
- Confusion matrices
- Classification reports
- Model comparison charts
- Sample predictions with visual feedback

## 🔬 Key Concepts

### Quantum Kernel

A quantum kernel computes the similarity between two feature vectors by:
1. Encoding features into quantum states using parameterized quantum circuits
2. Measuring the overlap between quantum states
3. Optimizing circuit parameters to maximize kernel-target alignment

### Kernel Target Alignment (KTA)

An optimization technique that aligns the quantum kernel matrix with the ideal kernel based on training labels, improving classification performance.

### Hybrid Approach

By combining classical and quantum computing:
- **Classical CNN** handles complex, high-dimensional raw data (images)
- **Quantum Kernel** operates on reduced feature space where quantum advantage may emerge

## 📝 References

- **Undergraduate Thesis (TFG)**: [Aprendizaje Automático Mediante Computación Cuántica - UVaDocs](https://uvadoc.uva.es/handle/10324/63011)
- PennyLane: [https://pennylane.ai/](https://pennylane.ai/)
- Quantum Machine Learning: [https://pennylane.ai/qml/](https://pennylane.ai/qml/)
- Dogs vs Cats Dataset: [https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)

---

**Note**: Quantum computing simulations can be computationally expensive. The code includes optimizations like sample size limits and batch processing to manage computational costs. For production use, consider cloud-based quantum computing services or quantum hardware access.