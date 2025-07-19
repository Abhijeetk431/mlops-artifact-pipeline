# tests/test_train.py
import pytest
import json
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from train import load_config, train_model

CONFIG_PATH = 'config/config.json'
MODEL_PATH = 'models/model_train.pkl'

@pytest.fixture(scope="module")
def sample_config():
    """Fixture to create a temporary config.json for testing."""
    config_data = {
        "C": 0.1,
        "solver": "liblinear",
        "max_iter": 1000
    }
    # Ensure config directory exists
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config_data, f)
    yield config_data
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)

@pytest.fixture(scope="module")
def digits_dataset():
    """Fixture to load the digits dataset."""
    digits = load_digits()
    X, y = digits.data, digits.target
    return X, y

# (a) Configuration File Loading Tests
def test_config_loads_successfully(sample_config):
    """Test that the configuration file loads successfully."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict)
    assert config == sample_config

def test_required_hyperparameters_exist(sample_config):
    """Check that all required hyperparameters exist in the configuration."""
    config = load_config(CONFIG_PATH)
    assert 'C' in config
    assert 'solver' in config
    assert 'max_iter' in config

def test_hyperparameter_data_types(sample_config):
    """Check that the values have the correct data types."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config['C'], float)
    assert isinstance(config['solver'], str)
    assert isinstance(config['max_iter'], int)

# (b) Model Creation Tests
def test_model_creation_returns_logistic_regression_object(digits_dataset, sample_config):
    """Verify that the training function returns a LogisticRegression object."""
    X, y = digits_dataset

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    config = load_config(CONFIG_PATH)
    model = train_model(X_train, y_train, config)
    assert isinstance(model, LogisticRegression)

def test_model_is_fitted(digits_dataset, sample_config):
    """Optionally, confirm the object has been fitted (e.g., by checking attributes like coef_ or classes_)."""
    X, y = digits_dataset
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    config = load_config(CONFIG_PATH)
    model = train_model(X_train, y_train, config)

    assert hasattr(model, 'coef_')
    assert model.coef_ is not None

    assert hasattr(model, 'classes_')
    assert model.classes_ is not None

# (c) Model Accuracy Test
def test_model_accuracy_above_threshold(digits_dataset, sample_config):
    """Check that the accuracy is above some threshold to verify correctness of training logic."""
    X, y = digits_dataset

    # Train and test on full dataset to simulate in-sample accuracy
    config = load_config(CONFIG_PATH)
    model = train_model(X, y, config)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    expected_min_accuracy = 0.95
    print(f"\nDebug: In-sample accuracy in test: {accuracy:.4f}")
    assert accuracy > expected_min_accuracy