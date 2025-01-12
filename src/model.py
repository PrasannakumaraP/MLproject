"""
Module for defining and training the model.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Train Model
def train_model(x_train, y_train):
    """Train the model."""
    model = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth = 3)
    model.fit(x_train, y_train)
    return model

# Evaluate model"""
def evaluate_model(model, x_test, y_test):
    """Evaluate the model using accuracy."""
    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    # Calculate accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    return accuracy
