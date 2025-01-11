"""
Module for data preprocessing.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(path):
    """Load data from the specified path and split into train/test sets."""
    data = pd.read_csv(path)
    x = data.drop(columns=['id', 'purchase'])  # Adjust as needed
    y = data['purchase']  # Target variable

    # One-hot encode categorical variables
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first'), ['education_level', 'marital_status'])
        ],
        remainder='passthrough'  # Keep other columns as is
    )

    x_transformed = column_transformer.fit_transform(x)

    return train_test_split(x_transformed, y, test_size=0.2, random_state=42)
