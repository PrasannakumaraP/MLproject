import joblib
from preprocess import load_data
from model import evaluate_model

DATA_PATH = "data/raw/sample_data.csv"  # Adjust this path accordingly

def test_model_accuracy():
    """Test the accuracy of the model."""
    x_train, x_test, y_train, y_test = load_data(DATA_PATH)
    model = joblib.load("models/model.pkl")
    accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {accuracy}")
    assert accuracy >= 0.75  # Assuming your model should have at least 75% accuracy
