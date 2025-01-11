from preprocess import load_data
from model import train_model, evaluate_model

data_path = "data/raw/sample_data.csv"

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data(data_path)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Save the model
    import joblib
    joblib.dump(model, "models/model.pkl")
