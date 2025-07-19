# src/inference.py
import joblib
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
import os

def load_model(model_path):
    """Loads a trained model from a .pkl file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def make_predictions(model, X):
    """Generates predictions using the loaded model."""
    print("Generating predictions...")
    predictions = model.predict(X)
    print("Predictions generated.")
    return predictions

def main():
    print("Starting model inference pipeline...")

    model_path = 'trained-model/model_train.pkl'

    # 1. Load the trained model
    try:
        model = load_model(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'trained-model' artifact was downloaded correctly and contains 'model_train.pkl'.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # 2. Load the digit classification dataset
    digits = load_digits()
    X, y_true = digits.data, digits.target
    print(f"Dataset loaded for inference: {X.shape[0]} samples, {X.shape[1]} features.")

    # 3. Generate predictions
    y_pred = make_predictions(model, X)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\nInference Performance on Dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("Model inference pipeline finished.")

if __name__ == "__main__":
    main()