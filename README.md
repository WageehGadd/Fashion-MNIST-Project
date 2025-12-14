# Fashion-MNIST Image Classification

A deep learning project for classifying fashion items using Convolutional Neural Networks (CNN) on the Fashion-MNIST dataset.

## Project Overview

This project implements a custom CNN architecture to classify grayscale images of clothing items into 10 categories. The model achieves **94.87% accuracy** on the test set, demonstrating strong performance in distinguishing between similar fashion items.

### Dataset

**Fashion-MNIST** is a dataset of Zalando's article images consisting of:
- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 categories

| Label | Category |
|-------|----------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Project Structure

```
project/
├── code/
│   ├── pre-train-model.py   # Alternative ResNet50V2 implementation
│   ├── dataset.py           # Data loading and preprocessing
│   ├── evaluate.py          # Model evaluation and visualization
│   ├── main.py              # Main pipeline orchestrator
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training loop and callbacks
│   └── utils.py             # Configuration and utilities
├── data/
│   ├── fashion-mnist_test.csv       # Test dataset (10,000 samples)
│   ├── fashion-mnist_train.csv      # Training dataset (60,000 samples)
│   ├── t10k-images-idx3-ubyte       # Test images (binary format)
│   ├── t10k-labels-idx1-ubyte       # Test labels (binary format)
│   ├── train-images-idx3-ubyte      # Training images (binary format)
│   └── train-labels-idx1-ubyte      # Training labels (binary format)
├── Reports/
│   └── pre-Train Model Report.pdf    # Alternative model documentation
├── results/
│   ├── accuracy_curve.png           # Separate accuracy visualization
│   ├── classification_report.txt    # Per-class performance metrics
│   ├── confusion_matrix.png         # Confusion matrix heatmap
│   ├── loss_curve.png               # Separate loss visualization
│   ├── sample_predictions_analysis.txt  # Sample predictions breakdown
│   ├── training_curves_1.png        # Basic training curves
│   └── training_curves_2.png        # Enhanced training curves
├── saved_model/
│   ├── best_model.h5           # Best model checkpoint
│   └── training_history.pkl    # Training history object
├── .gitignore             
├── README.md              
└── requirements.txt        
```


## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (optional, but recommended)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fashion-mnist-classification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**

Place the Fashion-MNIST CSV files in the `data/` directory:
- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

### Usage

#### Training the Model

Run the complete training pipeline:

```bash
python code/main.py
```

This will:
- Load and preprocess the dataset
- Split data into train/validation sets (90/10)
- Build the CNN architecture
- Train the model with data augmentation
- Save the best model checkpoint

#### Evaluating the Model

After training, evaluate the model performance:

```bash
python code/evaluate.py
```

This generates:
- Training curves (accuracy and loss)
- Confusion matrix
- Classification report
- Sample predictions analysis

## Model Architecture

The custom CNN consists of three convolutional blocks with increasing complexity:

```
Input (28×28×1)
    ↓
Conv Block 1: 2× Conv2D(64) + BatchNorm + ReLU → MaxPool → Dropout(0.25)
    ↓
Conv Block 2: 2× Conv2D(128) + BatchNorm + ReLU → MaxPool → Dropout(0.3)
    ↓
Conv Block 3: 2× Conv2D(256) + BatchNorm + ReLU → GlobalAvgPool → Dropout(0.4)
    ↓
Dense Block: Dense(256) + BatchNorm + ReLU → Dropout(0.4)
    ↓
Output: Dense(10, softmax)
```

**Key Features:**
- Batch normalization for training stability
- Dropout for regularization
- Global average pooling to reduce parameters
- Progressive channel expansion (64 → 128 → 256)

## Configuration

Key hyperparameters in `utils.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `PATIENCE` | 10 | Early stopping patience |
| `IMG_SHAPE` | (28, 28, 1) | Input image dimensions |
| `NUM_CLASSES` | 10 | Number of output classes |
| `SEED` | 42 | Random seed for reproducibility |

## Training Strategy

### Data Augmentation

To improve generalization, the following augmentation techniques are applied:

- Rotation: ±10°
- Width/Height shift: 10%
- Shear transformation: 10%
- Zoom: ±10%
- Horizontal flip

### Optimization

- **Optimizer**: Adam (learning rate: 1e-3)
- **Loss function**: Categorical cross-entropy
- **Learning rate schedule**: ReduceLROnPlateau (factor: 0.5, patience: 3)
- **Early stopping**: Patience of 10 epochs on validation loss

### Callbacks

1. **ModelCheckpoint**: Saves best model based on validation loss
2. **EarlyStopping**: Stops training if no improvement
3. **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

## Results

### Model Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 94.87% |
| **Best Validation Accuracy** | 94.18% (Epoch 23) |
| **Training Time** | ~33 epochs (early stopped) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| T-shirt/top | 0.91 | 0.90 | 0.90 |
| Trouser | 0.99 | 1.00 | 1.00 |
| Pullover | 0.95 | 0.92 | 0.93 |
| Dress | 0.93 | 0.96 | 0.95 |
| Coat | 0.94 | 0.91 | 0.92 |
| Sandal | 0.99 | 0.99 | 0.99 |
| Shirt | 0.83 | 0.86 | 0.84 |
| Sneaker | 0.97 | 0.98 | 0.97 |
| Bag | 1.00 | 1.00 | 1.00 |
| Ankle boot | 0.98 | 0.98 | 0.98 |

### Key Observations

- **Best performers**: Bag (100%), Trouser (99.5%), Sandal (99%)
- **Most challenging**: Shirt (84%) - often confused with T-shirt/top, Pullover, and Coat
- **Common confusion**: Pullovers vs Shirts, Coats vs other upper-body garments

## Output Files

After running the pipeline, the following files are generated:

### Model Files (`saved_model/`)
- `best_model.h5` - Trained model weights
- `training_history.pkl` - Training history object

### Evaluation Results (`results/`)
- `training_curves_1.png` - Basic accuracy/loss plots
- `training_curves_2.png` - Enhanced training curves with annotations
- `accuracy_curve.png` - Detailed accuracy visualization
- `loss_curve.png` - Detailed loss visualization
- `confusion_matrix.png` - Visual confusion matrix
- `classification_report.txt` - Detailed per-class metrics
- `sample_predictions_analysis.txt` - Analysis of 20 sample predictions

## Advanced Usage

### Transfer Learning Alternative

The project includes an experimental ResNet50V2-based approach in `pre-train-model.py`:

```bash
python code/pre-train-model.py
```

This uses transfer learning with ImageNet weights, requiring image resizing to 85×85×3.


## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- TensorFlow/Keras framework
- scikit-learn for evaluation metrics

---

**Note**: This project is designed for educational purposes as a final project for the CS 417 Neural Networks course. It demonstrates best practices in deep learning project structure, documentation, and evaluation.
