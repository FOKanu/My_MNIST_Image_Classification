#!/usr/bin/env python3
"""
Main script for MNIST Classification Project
This script demonstrates the complete pipeline from data loading to model evaluation.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from src.data_preprocessing import MNISTDataPreprocessor, display_sample_images
from src.model import MNISTCNN
from src.evaluation import MNISTEvaluator

import numpy as np
import matplotlib.pyplot as plt

def main():
    """
    Main function that runs the complete MNIST classification pipeline.
    """
    print("="*60)
    print("MNIST CLASSIFICATION PROJECT")
    print("="*60)

    # Step 1: Data Preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = MNISTDataPreprocessor()
    data = preprocessor.preprocess_all()

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    y_train_cat = data['y_train_cat']
    y_test_cat = data['y_test_cat']

    # Step 2: Model Creation
    print("\n2. Creating CNN model...")
    cnn_model = MNISTCNN(input_shape=(28, 28, 1), num_classes=10)
    model = cnn_model.build_model()

    # Step 3: Model Training
    print("\n3. Training model...")
    history = cnn_model.train(
        X_train=X_train,
        y_train=y_train_cat,
        epochs=5,
        batch_size=32,
        validation_split=0.3,
        early_stopping=True,
        model_checkpoint=True
    )

    # Step 4: Model Evaluation
    print("\n4. Evaluating model...")
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test_cat)

    # Step 5: Comprehensive Analysis
    print("\n5. Running comprehensive evaluation...")
    evaluator = MNISTEvaluator(model, X_test, y_test, y_test_cat)
    evaluation_results = evaluator.comprehensive_evaluation(history=history)

    # Step 6: Save Model
    print("\n6. Saving model...")
    os.makedirs('models', exist_ok=True)
    cnn_model.save_model('models/mnist_cnn_final.h5')

    # Step 7: Final Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Model Parameters: {model.count_params():,}")
    print("="*60)

    print("\n✅ Project completed successfully!")
    print("📁 Model saved to: models/mnist_cnn_final.h5")
    print("📊 Check the generated plots for detailed analysis.")

if __name__ == "__main__":
    main()
