"""
Module for data preprocessing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    """Load data from the specified path and split into train/test sets."""
    data = pd.read_csv(path)
    x = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    return train_test_split(x, y, test_size=0.2, random_state=42)
