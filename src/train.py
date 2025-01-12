"""
Module for training the model.
"""
import pickle
from joblib import dump
from preprocess import load_data
from model import train_model, evaluate_model

DATA_PATH = "data/raw/temps.csv"  # Constant name updated to UPPER_CASE

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)

    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the model
    # Save the model using pickle
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    # Save the model using Joblib
    dump(model, 'models/model_j.joblib')
