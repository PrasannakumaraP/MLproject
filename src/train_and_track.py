"""
Module for defining, training, and evaluating a Random Forest model.
This module loads data, preprocesses it, trains a model, evaluates its performance,
and saves the trained model to the 'models' directory.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
import mlflow

def load_data(path):
    """
    Load data from a CSV file, preprocess it, and split into training and testing sets.

    Args:
        path (str): The file path to the CSV data file.

    Returns:
        tuple: Four NumPy arrays: train_x, test_x, train_y, test_y
    """
    data = pd.read_csv(path)
    print(data.head())

    features = pd.get_dummies(data)
    x_label = features.drop('actual', axis=1)
    x_label = np.array(x_label)
    y_label = np.array(features['actual'])
    return train_test_split(x_label, y_label, test_size=0.25, random_state=42)

def train_model(train_x, train_y, x_test_lbl, y_test_lbl):
    """
    Train a Random Forest Regressor model using the provided training data.

    Args:
        train_x (np.ndarray): Training feature set.
        train_y (np.ndarray): Training labels.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    model_instance = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=3)
    model_instance.fit(train_x, train_y)

    # Evaluate the trained model for Experiment tracking using MLflow
    pred_res = model_instance.predict(x_test_lbl)

    errors = abs(pred_res - y_test_lbl)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / y_test_lbl)
    mse = mean_squared_error(y_test_lbl, pred_res)
    rmse = mse ** 0.5
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    accuracy_traking = 100 - np.mean(mape)
    print(f"Model accuracy: {accuracy_traking}")

    # Start MLflow logging
    mlflow.start_run() # pylint: disable=E1101

    # Log hyperparameters
    mlflow.log_param("n_estimators", 1000) # pylint: disable=E1101
    mlflow.log_param("random_state", 42)   # pylint: disable=E1101

    # Log metrics
    mlflow.log_metric("accuracy", accuracy_traking)  # pylint: disable=E1101

    # Log the trained model
    mlflow.sklearn.log_model(model_instance, "model") # pylint: disable=E1101

    # End MLflow run
    mlflow.end_run() # pylint: disable=E1101
    return model_instance

def evaluate_model(model_instance, test_x, test_y):
    """
    Evaluate the trained model using test data and print performance metrics.

    Args:
        model_instance (RandomForestRegressor): Trained model to evaluate.
        test_x (np.ndarray): Test feature set.
        test_y (np.ndarray): Test labels.

    Returns:
        float: Accuracy of the model.
    """
    predictions = model_instance.predict(test_x)
    errors = abs(predictions - test_y)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / test_y)
    mse = mean_squared_error(test_y, predictions)
    rmse = mse ** 0.5
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    accuracy_value = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy_value, 2), '%.')
    return accuracy_value

if __name__ == "__main__":
    # Main script to load data, train the model, evaluate it, and save the trained model.
    DATA_PATH = "data/raw/temps.csv"
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)

    model = train_model(x_train, y_train,x_test,y_test)
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the model
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    dump(model, 'models/model_j.joblib')
