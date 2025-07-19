# src/train.py
import json
import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

def load_config(config_path='config/config.json'):
    """Loads hyperparameters from a JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def train_model(X_train, y_train, config):
    """Trains a Logistic Regression model with given hyperparameters."""
    model = LogisticRegression(
        C=config['C'],
        solver=config['solver'],
        max_iter=config['max_iter'],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    print("Starting model training pipeline...")

    # 1. Load the digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # 2. Read hyperparameters from config/config.json
    try:
        config = load_config()
        print(f"Configuration loaded: C={config['C']}, solver={config['solver']}, max_iter={config['max_iter']}")
    except FileNotFoundError:
        print("Error: config/config.json not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: config/config.json is not valid JSON.")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing key in config.json: {e}. Ensure C, solver, and max_iter are present.")
        exit(1)

    # 3. Train a LogisticRegression model
    model = train_model(X_train, y_train, config)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nModel Performance on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/model_train.pkl'
    joblib.dump(model, model_path) # Save the model 
    print(f"Trained model saved to {model_path}")
    print("Model training pipeline finished.")

if __name__ == "__main__":
    main()