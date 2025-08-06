"""
Evaluation utilities for MNIST model.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


class MNISTEvaluator:
    """Handles evaluation and visualization of MNIST model performance."""

    def __init__(self, model, X_test, y_test, y_test_cat):
        """Initialize the evaluator.

        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels (integers)
            y_test_cat: Test labels (one-hot encoded)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_cat = y_test_cat
        self.predictions = None
        self.predicted_classes = None
        self.confusion_matrix = None

    def get_predictions(self):
        """Get model predictions on test data.

        Returns:
            tuple: (predictions, predicted_classes)
        """
        print("Generating predictions...")
        self.predictions = self.model.predict(self.X_test)
        self.predicted_classes = np.argmax(self.predictions, axis=1)

        # Calculate confusion matrix
        self.confusion_matrix = confusion_matrix(self.y_test, self.predicted_classes)

        return self.predictions, self.predicted_classes

    def plot_training_history(self, history):
        """Plot training history.

        Args:
            history: Training history object
        """
        plt.figure(figsize=(12, 4))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        if self.predicted_classes is None:
            self.get_predictions()

        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_class_accuracy(self):
        """Plot per-class accuracy."""
        if self.confusion_matrix is None:
            self.get_predictions()

        class_accuracy = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)

        plt.figure(figsize=(10, 6))
        plt.bar(range(10), class_accuracy)
        plt.title('Per-Class Accuracy')
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, acc in enumerate(class_accuracy):
            plt.text(i, acc, f'{acc:.3f}', ha='center', va='bottom')
        plt.show()

    def plot_error_analysis(self, num_samples=10):
        """Plot misclassified examples.

        Args:
            num_samples: Number of examples to show
        """
        if self.predicted_classes is None:
            self.get_predictions()

        errors = self.predicted_classes != self.y_test
        error_indices = np.where(errors)[0]

        if len(error_indices) == 0:
            print("No errors found!")
            return

        samples = np.random.choice(error_indices,
                                 size=min(num_samples, len(error_indices)),
                                 replace=False)

        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(samples):
            plt.subplot(1, num_samples, i+1)
            # Get the image
            img = self.X_test[idx]
            # Convert tensor to numpy if needed
            if tf.is_tensor(img):
                img = img.numpy()
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'True: {self.y_test[idx]}\nPred: {self.predicted_classes[idx]}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_sample_predictions(self, num_samples=10):
        """Plot sample predictions with confidence scores.

        Args:
            num_samples: Number of samples to show
        """
        if self.predictions is None:
            self.get_predictions()

        indices = np.random.choice(len(self.X_test), num_samples, replace=False)

        plt.figure(figsize=(15, 3))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i+1)
            # Get the image
            img = self.X_test[idx]
            # Convert tensor to numpy if needed
            if tf.is_tensor(img):
                img = img.numpy()
            plt.imshow(img.squeeze(), cmap='gray')

            pred = self.predicted_classes[idx]
            conf = self.predictions[idx][pred]
            color = 'green' if pred == self.y_test[idx] else 'red'

            plt.title(f'Pred: {pred}\nConf: {conf:.2f}', color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def comprehensive_evaluation(self, history=None):
        """Run comprehensive evaluation and visualization.

        Args:
            history: Training history object (optional)
        """
        print("Running comprehensive evaluation...")

        # Get predictions if not already done
        if self.predictions is None:
            self.get_predictions()

        # Plot training history if provided
        if history is not None:
            self.plot_training_history(history)

        # Print classification report
        self.print_classification_report()

        # Plot confusion matrix
        self.plot_confusion_matrix()

        # Plot class accuracy
        self.plot_class_accuracy()

        # Plot error analysis
        self.plot_error_analysis()

        # Plot sample predictions
        self.plot_sample_predictions()

        return {
            'predictions': self.predictions,
            'predicted_classes': self.predicted_classes,
            'accuracy': np.mean(self.predicted_classes == self.y_test),
            'confusion_matrix': self.confusion_matrix
        }

    def print_classification_report(self):
        """Print detailed classification report."""
        if self.predicted_classes is None:
            self.get_predictions()

        print("\n" + "="*50)
        print("CLASSIFICATION REPORT")
        print("="*50)

        # Create label names
        labels = [f"Digit {i}" for i in range(10)]

        print(classification_report(self.y_test, self.predicted_classes,
                                 target_names=labels))

    def print_summary_statistics(self):
        """Print summary statistics."""
        if self.predicted_classes is None:
            self.get_predictions()

        accuracy = np.mean(self.predicted_classes == self.y_test)
        errors = np.sum(self.predicted_classes != self.y_test)

        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Total samples: {len(self.y_test)}")
        print(f"Correct predictions: {len(self.y_test) - errors}")
        print(f"Incorrect predictions: {errors}")
        print(f"Accuracy: {accuracy:.4f}")
        print("="*50)
