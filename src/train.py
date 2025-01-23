import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump

# Load data and preprocess
def load_data(path):
    data = pd.read_csv(path)
    print(data.head())

    features = pd.get_dummies(data)
    x_label = features.drop('actual', axis=1)
    x_label = np.array(x_label)
    y_label = np.array(features['actual'])
    return train_test_split(x_label, y_label, test_size=0.25, random_state=42)

# Train Model
def train_model(train_x, train_y):
    model_instance = RandomForestRegressor(n_estimators=1000, random_state=42, max_depth=3)
    model_instance.fit(train_x, train_y)
    return model_instance

# Evaluate model
def evaluate_model(model_instance, test_x, test_y):
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

# Main execution block
if __name__ == "__main__":
    DATA_PATH = "data/raw/temps.csv"
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)

    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save the model
    with open('models/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    dump(model, 'models/model_j.joblib')
