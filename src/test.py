"""
Module for testing the model.
"""

import joblib
from preprocess import load_data
from model import evaluate_model

DATA_PATH = "temps.csv"  # Constant name updated to UPPER_CASE

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)
    model = joblib.load("model.pkl")
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {accuracy}")