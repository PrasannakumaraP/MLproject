"""
Module for testing the model.
"""
import joblib
import pytest
from preprocess import load_data
from model import evaluate_model

DATA_PATH = "data/raw/temps.csv"  # Constant name updated to UPPER_CASE

def test_model_accuracy():
    """Test the accuracy of the loaded model."""
    # Load the data
    _, x_test, _, y_test = load_data(DATA_PATH)

    # Load the trained model
    model = joblib.load("models/model.pkl")

    # Evaluate the model
    accuracy = evaluate_model(model, x_test, y_test)

    # Assert the accuracy is above a certain threshold
    assert accuracy >= 0.7, "Model accuracy is below the expected threshold"

#Test script
if __name__ == "__main__":
    pytest.main()  # Run pytest when executing the script directly
