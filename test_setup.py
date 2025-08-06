#!/usr/bin/env python3
"""
Test script to verify the MNIST Classification project setup.
This script tests that all imports work and basic functionality is available.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        # Test standard libraries
        import numpy as np
        import matplotlib.pyplot as plt
        import tensorflow as tf
        print("✅ Standard libraries imported successfully")

        # Test our custom modules
        sys.path.append('src')
        from src.data_preprocessing import MNISTDataPreprocessor
        from src.model import MNISTCNN
        from src.evaluation import MNISTEvaluator
        print("✅ Custom modules imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\nTesting data preprocessing...")

    try:
        preprocessor = MNISTDataPreprocessor()
        print("✅ Data preprocessor created successfully")

        # Test data loading (this will download MNIST if not available)
        print("Loading MNIST dataset (this may take a moment)...")
        preprocessor.load_data()
        print("✅ MNIST dataset loaded successfully")

        return True

    except Exception as e:
        print(f"❌ Data preprocessing error: {e}")
        return False

def test_model_creation():
    """Test model creation functionality."""
    print("\nTesting model creation...")

    try:
        cnn_model = MNISTCNN(input_shape=(28, 28, 1), num_classes=10)
        model = cnn_model.build_model()
        print("✅ CNN model created successfully")

        # Test model summary
        print(f"Model parameters: {model.count_params():,}")

        return True

    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("MNIST CLASSIFICATION PROJECT - SETUP TEST")
    print("="*50)

    tests = [
        ("Import Test", test_imports),
        ("Data Preprocessing Test", test_data_preprocessing),
        ("Model Creation Test", test_model_creation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Run the notebook: jupyter notebook notebooks/mnist_classification.ipynb")
        print("2. Or run the main script: python main.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

    print("="*50)

if __name__ == "__main__":
    main()
