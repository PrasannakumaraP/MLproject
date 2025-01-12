"""
Module for data preprocessing.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load and preprocess data, ensuring all features are numeric."""
    data = pd.read_csv(path)
    print(data.head())

    features = pd.get_dummies(data)
    features.head(5)
    X_label= features.drop('actual', axis = 1) # axis 1 refers to the columns
    X_label = np.array(X_label)
    y_label = np.array(features['actual'])
    print("\nFeatures and Target:")
    print(X_label[:5])
    print(y_label[:5])
    # Split into train and test sets
    return train_test_split(X_label, y_label, test_size=0.25, random_state=42)

# Debugging function call
try:
    x_train, x_test, y_train, y_test = load_data("temps.csv")
    print("\nData loaded successfully!")
except Exception as e:
    print(f"Error: {e}")
