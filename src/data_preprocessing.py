"""
Data preprocessing utilities for MNIST dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class MNISTDataPreprocessor:
    """Handles loading and preprocessing of MNIST dataset."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_cat = None
        self.y_test_cat = None

    def load_data(self):
        """Load MNIST dataset."""
        print("Loading MNIST dataset...")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.mnist.load_data()
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        return self

    def normalize_data(self):
        """Normalize pixel values to [0,1] range."""
        print("Normalizing pixel values...")
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        print("Data normalization completed.")
        return self

    def reshape_for_cnn(self):
        """Reshape data for CNN input (add channel dimension)."""
        print("Reshaping data for CNN input...")
        self.X_train = np.expand_dims(self.X_train, axis=-1)
        self.X_test = np.expand_dims(self.X_test, axis=-1)
        print(f"Training data shape after reshape: {self.X_train.shape}")
        print(f"Test data shape after reshape: {self.X_test.shape}")
        return self

    def encode_labels(self):
        """One-hot encode the target labels."""
        print("Encoding labels...")
        self.y_train_cat = to_categorical(self.y_train)
        self.y_test_cat = to_categorical(self.y_test)
        print(f"Training labels shape: {self.y_train_cat.shape}")
        print(f"Test labels shape: {self.y_test_cat.shape}")
        return self

    def preprocess_all(self):
        """Run complete preprocessing pipeline."""
        self.load_data()
        self.normalize_data()
        self.reshape_for_cnn()
        self.encode_labels()
        print("Data preprocessing completed successfully!")
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'y_train_cat': self.y_train_cat,
            'y_test_cat': self.y_test_cat
        }

    def get_data_info(self):
        """Print comprehensive dataset information."""
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Training samples: {len(self.y_train)}")
        print(f"Test samples: {len(self.y_test)}")
        print(f"Image dimensions: {self.X_train.shape[1:]} (height, width, channels)")
        print(f"Number of classes: {len(np.unique(self.y_train))}")

        print("Class distribution (training):")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples")
        print("="*50)


def display_sample_images(X, y, num_samples=10):
    """Display sample images from the dataset.

    Args:
        X: Image data (N, height, width, channels)
        y: Labels (N,)
        num_samples: Number of samples to display
    """
    indices = np.random.choice(len(X), num_samples, replace=False)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        # Convert tensor to numpy array if needed
        if isinstance(X[idx], tf.Tensor):
            img = X[idx].numpy()
        else:
            img = X[idx]
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'Label: {y[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
