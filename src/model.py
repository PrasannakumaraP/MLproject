"""
Module for defining and training the model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(x_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluate the model using accuracy."""
    predictions = model.predict(x_test)
    return accuracy_score(y_test, predictions)
