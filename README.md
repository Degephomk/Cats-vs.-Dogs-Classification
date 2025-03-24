# ResNet152 Transfer Learning: Cats vs. Dogs Classification

## Overview
This project implements a deep learning model for classifying images of cats and dogs using transfer learning with ResNet152. By leveraging advanced techniques like stochastic depth, cosine learning rate decay, and no bias decay, the model achieves an impressive **99.78% accuracy** on the test set.

## Features
- **Transfer Learning**: Fine-tunes a pre-trained ResNet152 model from torchvision.
- **Stochastic Depth**: Randomly drops layers during training to improve generalization.
- **Cosine Learning Rate Decay**: Gradually reduces the learning rate for stable convergence.
- **No Bias Decay**: Excludes bias from regularization for better optimization.
- **Custom Learning Rates**: Assigns different learning rates to layers for efficient training.

## Dataset
- **Cats vs. Dogs**: Sourced from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) (~25,000 images).
- Training and test sets are preprocessed with augmentations (resize, crop, flip, etc.).

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- OpenCV
- Matplotlib
- TQDM

Install dependencies:
```bash
pip install torch torchvision opencv-python matplotlib tqdm
```

## Usage
1. **Prepare the Dataset**:
   - Download the dataset and place it in `datasets/dogs_and_cats/` with `train/` and `test/` subfolders.
2. **Run the Code**:
   - Execute the script or notebook:
     ```bash
     python resnet_transfer_learning.py
     ```
     or open `resnet-transfer-learning-accuracy-99-78.ipynb` in Jupyter/Colab.
3. **Output**:
   - Training loss and accuracy plots are generated.
   - Final test accuracy: **99.78%**.

## Results
- **Accuracy**: 99.78% on the test set after 5 epochs.
- **Loss and Accuracy Plots**:


## Project Structure
- `resnet-transfer-learning-accuracy-99-78.ipynb`: Main notebook with code and visualizations.
- `datasets/`: Directory for the Cats vs. Dogs dataset (not included).
- `results/`: Output plots (loss and accuracy).

## How It Works
1. Loads pre-trained ResNet152 weights.
2. Modifies the architecture with stochastic depth and a new fully connected layer (2 classes).
3. Freezes early layers and fine-tunes later layers with custom learning rates.
4. Trains with AdamW optimizer and cosine annealing scheduler.

## License
MIT License - feel free to use and modify this code!



