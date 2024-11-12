import pytest
import pandas as pd
import numpy as np
from src.ml.models import ModelTrainer, ModelSelector

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)
    
    data = {
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(150, 250, n_samples),
        'low': np.random.uniform(90, 180, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.uniform(1000000, 5000000, n_samples),
        'target': np.random.uniform(100, 200, n_samples)
    }
    
    return pd.DataFrame(data, index=dates)

def test_model_trainer_init():
    """Test ModelTrainer initialization."""
    trainer = ModelTrainer()
    assert 'linear' in trainer.models
    assert 'random_forest' in trainer.models
    assert 'xgboost' in trainer.models

def test_model_preparation(sample_data):
    """Test data preparation in ModelTrainer."""
    trainer = ModelTrainer()
    X, y = trainer.prepare_data(sample_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert not np.any(np.isnan(X))