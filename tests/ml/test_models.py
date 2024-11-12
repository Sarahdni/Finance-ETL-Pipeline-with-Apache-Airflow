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

class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer()
        assert 'linear' in trainer.models
        assert 'random_forest' in trainer.models
        assert 'xgboost' in trainer.models
        assert hasattr(trainer, 'scaler')
    
    def test_prepare_data_success(self, sample_data):
        """Test successful data preparation."""
        trainer = ModelTrainer()
        X, y = trainer.prepare_data(sample_data)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert not np.any(np.isnan(X))
    
    def test_prepare_data_empty(self):
        """Test data preparation with empty DataFrame."""
        trainer = ModelTrainer()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty DataFrame provided"):
            trainer.prepare_data(empty_df)
    
    def test_train_models_success(self, sample_data):
        """Test successful model training."""
        trainer = ModelTrainer()
        results = trainer.train_models(sample_data)
        
        assert isinstance(results, dict)
        for model_name, model_info in results.items():
            assert 'model' in model_info
            assert 'metrics' in model_info
            assert isinstance(model_info['metrics'], dict)
            assert all(metric in model_info['metrics'] 
                      for metric in ['rmse', 'mae', 'r2'])

class TestModelSelector:
    """Test suite for ModelSelector class."""
    
    @pytest.fixture
    def trained_models(self, sample_data):
        """Fixture for trained models."""
        trainer = ModelTrainer()
        return trainer.train_models(sample_data)
    
    def test_select_best_model(self, trained_models):
        """Test best model selection."""
        selector = ModelSelector()
        best_model = selector.select_best_model(trained_models)
        
        assert isinstance(best_model, dict)
        assert 'model' in best_model
        assert 'metrics' in best_model
        assert 'feature_names' in best_model
    
    def test_generate_predictions(self, sample_data, trained_models):
        """Test prediction generation."""
        selector = ModelSelector()
        best_model = selector.select_best_model(trained_models)
        
        # Prepare test data
        test_features = sample_data.drop('target', axis=1).head(10)
        predictions = selector.generate_predictions(best_model, test_features)
        
        assert isinstance(predictions, dict)
        assert 'predictions' in predictions
        assert 'timestamp' in predictions
        assert len(predictions['predictions']) == len(test_features)
    
    def test_invalid_model_selection(self):
        """Test model selection with invalid input."""
        selector = ModelSelector()
        with pytest.raises(ValueError, match="No model results provided"):
            selector.select_best_model({})