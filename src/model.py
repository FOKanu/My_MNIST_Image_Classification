"""
CNN model for MNIST digit classification.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models


class MNISTCNN:
    """CNN model for MNIST digit classification."""

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """Initialize the CNN model.

        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """Build and compile the CNN model.

        Returns:
            Compiled Keras model
        """
        print("Building CNN architecture...")
        print("Model architecture:")

        # Create Sequential model
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(8, (4, 4), activation='relu', padding='same',
                         input_shape=self.input_shape, name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),

            # Second Convolutional Block
            layers.Conv2D(16, (3, 3), activation='relu', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),

            # Flatten layer
            layers.Flatten(name='flatten'),

            # Dense layers
            layers.Dense(10, activation='relu', name='dense1'),
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])

        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def get_model_info(self):
        """Print detailed model information."""
        if self.model is None:
            print("Model hasn't been built yet. Call build_model() first.")
            return

        print("\n" + "="*50)
        print("MODEL ARCHITECTURE INFORMATION")
        print("="*50)

        # Print basic information
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
        print(f"Input shape: {self.input_shape}")
        print(f"Output classes: {self.num_classes}")

        print("\nLayer Details:")

        # Use model.summary() to get layer information
        def print_layer_info(layer):
            """Helper function to print layer details"""
            config = layer.get_config()
            print(f"\n  Layer {layer.name}:")
            print(f"    Type: {layer.__class__.__name__}")

            # Print layer-specific details
            if isinstance(layer, layers.Conv2D):
                print(f"    Details: filters={config['filters']}, kernel={config['kernel_size']}")
            elif isinstance(layer, layers.MaxPooling2D):
                print(f"    Details: pool_size={config['pool_size']}")
            elif isinstance(layer, layers.Dense):
                print(f"    Details: units={config['units']}")
            elif isinstance(layer, layers.Flatten):
                print(f"    Details: shape transformation")

            # Get output shape from layer config
            if hasattr(layer, 'output_shape'):
                if isinstance(layer.output_shape, tuple):
                    shape_str = 'x'.join(str(dim) for dim in layer.output_shape[1:])
                    print(f"    Output shape: {shape_str}")

            print(f"    Parameters: {layer.count_params():,}")

        # Print information for each layer
        for layer in self.model.layers:
            print_layer_info(layer)

        print("\n" + "="*50)

    def train(self, X_train, y_train, epochs=10, batch_size=32,
             validation_split=0.2, early_stopping=True, model_checkpoint=True):
        """Train the model.

        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping
            model_checkpoint: Whether to save best model

        Returns:
            Training history
        """
        if self.model is None:
            print("Model hasn't been built yet. Call build_model() first.")
            return None

        print("Starting training for {} epochs...".format(epochs))
        print("Batch size:", batch_size)
        print("Training samples:", len(X_train))

        callbacks = []

        # Add early stopping if requested
        if early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                )
            )

        # Add model checkpoint if requested
        if model_checkpoint:
            os.makedirs('../models', exist_ok=True)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    '../models/best_mnist_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.

        Args:
            X_test: Test data
            y_test: Test labels (one-hot encoded)

        Returns:
            tuple: (test_loss, test_accuracy)
        """
        if self.model is None:
            print("Model hasn't been built yet. Call build_model() first.")
            return None

        print("Evaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        return test_loss, test_accuracy

    def predict(self, X):
        """Make predictions on new data.

        Args:
            X: Input data

        Returns:
            Model predictions
        """
        if self.model is None:
            print("Model hasn't been built yet. Call build_model() first.")
            return None

        return self.model.predict(X)

    def save_model(self, filepath):
        """Save the model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            print("Model hasn't been built yet. Call build_model() first.")
            return

        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved model from disk.

        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
