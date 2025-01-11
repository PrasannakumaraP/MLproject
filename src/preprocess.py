import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    return train_test_split(X, y, test_size=0.2, random_state=42)
