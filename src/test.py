"""
Unit tests for the machine learning pipeline.

This file contains test cases to validate data preprocessing,
model training, and evaluation functions.
"""

import joblib
from preprocess import load_data
from model import evaluate_model

DATA_PATH = "data/raw/sample_data.csv"  # Adjust this path accordingly

def test_model_accuracy():
    """Test the accuracy of the model."""
    _, x_test, _, y_test = load_data(DATA_PATH)
    model = joblib.load("models/model.pkl")
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    assert accuracy >= 0.75  # Assuming your model should have at least 75% accuracy

if __name__ == "__main__":
    test_model_accuracy()
