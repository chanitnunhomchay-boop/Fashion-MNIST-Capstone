# Fashion-MNIST Image Classification

A deep learning project that classifies clothing images into 10 categories using a Convolutional Neural Network (CNN) built with PyTorch.

## 1. Project Overview

This project uses a CNN trained from scratch to classify grayscale clothing images from the Fashion-MNIST dataset.

**Objectives:**
- Build and train a CNN using PyTorch.
- Evaluate model performance on unseen test data.
- Analyze classification errors and model limitations.
- Provide reproducible training and evaluation code.

## 2. Dataset

Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

The dataset contains 70,000 grayscale images across 10 clothing categories.

| Dataset | Number of Images |
|---|---:|
| Training | 48,000 |
| Validation | 12,000 |
| Test | 10,000 |
| **Total** | **70,000** |

- Image size: 28 × 28 pixels
- Number of classes: 10
- Training and validation data are split from the original training set using random seed 42.
- The official test set is kept separate and is not used during training.

## 3. Model Architecture

The project uses a Convolutional Neural Network (CNN).

Architecture:
1. Input: 1 × 28 × 28
2. Convolutional layer: 1 → 32 channels
3. ReLU activation and Max Pooling
4. Convolutional layer: 32 → 64 channels
5. ReLU activation and Max Pooling
6. Flatten
7. Fully connected layer: 3136 → 128
8. Dropout: 0.3
9. Output layer: 128 → 10 classes

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Epochs | 10 |
| Random Seed | 42 |
| Dropout | 0.3 |

The best model checkpoint is selected based on validation accuracy.

## 5. Project Structure

```text
Fashion-MNIST-Capstone/
├── Fashion_MNIST_Capstone.ipynb
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── train.py
├── checkpoints/
├── evaluate.py
├── README.md
├── requirements.txt
└── .gitignore
```

The `checkpoints/` directory is generated during training and is not included in Git.

## 6. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Fashion-MNIST-Capstone
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows CMD:

```cmd
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. Training

Run the training script from the project root:

```bash
python -m src.train
```

The dataset will be downloaded automatically if it is not already available.

The best checkpoint will be saved to:

```text
checkpoints/best_model.pth
```

## 8. Evaluation

After training, run:

```bash
python evaluate.py
```

The evaluation script reports:
- Test Loss
- Test Accuracy
- Precision, Recall, and F1-score
- Confusion Matrix

## 9. Results

Final test results from the best validation checkpoint will be added after evaluation.

## 10. Error Analysis

The model has difficulty classifying the Shirt category compared to several other clothing categories.

Shirt images are frequently confused with T-shirt, Pullover, and Coat. This may be caused by similarities in their shapes and visual features, especially at the low resolution of 28 × 28 pixels.

## 11. Limitations and Future Improvements

- The model may confuse visually similar clothing categories.
- The dataset contains small grayscale images, limiting available visual details.
- Future improvements could include data augmentation, batch normalization, and architecture tuning.

## 12. Reproducibility

- Random seed: 42
- Training/validation split: 48,000 / 12,000
- Official test set: 10,000 images
- Hyperparameters and model architecture are defined in the source code.
- Training and evaluation can be reproduced using the provided scripts.