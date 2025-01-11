"""
Module for training the model.
"""

from preprocess import load_data
from model import train_model, evaluate_model
import joblib

DATA_PATH = "data/raw/sample_data.csv"  # Constant name updated to UPPER_CASE

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the model
    joblib.dump(model, "models/model.pkl")
