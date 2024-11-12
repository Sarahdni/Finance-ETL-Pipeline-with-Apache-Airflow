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
        'sma_20': np.random.uniform(100, 200, n_samples),
        'ema_12': np.random.uniform(100, 200, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.uniform(-10, 10, n_samples),
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
    
    def test_prepare_data(self, sample_data):
        """Test data preparation method."""
        trainer = ModelTrainer()
        X, y = trainer.prepare_data(sample_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))
    
    def test_prepare_data_empty_df(self):
        """Test data preparation with empty DataFrame."""
        trainer = ModelTrainer()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Empty DataFrame provided"):
            trainer.prepare_data(empty_df)
    
    def test_prepare_data_missing_target(self, sample_data):
        """Test data preparation with missing target column."""
        trainer = ModelTrainer()
        df_no_target = sample_data.drop('target', axis=1)
        
        with pytest.raises(ValueError, match="Target column not found"):
            trainer.prepare_data(df_no_target)
    
    def test_train_models(self, sample_data):
        """Test model training process."""
        trainer = ModelTrainer()
        results = trainer.train_models(sample_data)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        for model_name, model_info in results.items():
            assert 'model' in model_info
            assert 'metrics' in model_info
            assert 'scaler' in model_info
            assert 'feature_names' in model_info
            
            metrics = model_info['metrics']
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            
            assert metrics['rmse'] >= 0
            assert metrics['mae'] >= 0
            assert metrics['r2'] <= 1

class TestModelSelector:
    """Test suite for ModelSelector class."""
    
    @pytest.fixture
    def sample_model_results(self, sample_data):
        """Create sample model results for testing."""
        trainer = ModelTrainer()
        return trainer.train_models(sample_data)
    
    def test_select_best_model(self, sample_model_results):
        """Test best model selection."""
        selector = ModelSelector()
        best_model = selector.select_best_model(sample_model_results)
        
        assert isinstance(best_model, dict)
        assert 'model' in best_model
        assert 'metrics' in best_model
        assert 'scaler' in best_model
        assert 'feature_names' in best_model
    
    def test_select_best_model_empty_results(self):
        """Test best model selection with empty results."""
        selector = ModelSelector()
        
        with pytest.raises(ValueError, match="No model results provided"):
            selector.select_best_model({})
    
    def test_generate_predictions(self, sample_data, sample_model_results):
        """Test prediction generation."""
        selector = ModelSelector()
        best_model = selector.select_best_model(sample_model_results)
        
        # Use a subset of data for prediction testing
        test_features = sample_data.drop('target', axis=1).head(10)
        predictions = selector.generate_predictions(best_model, test_features)
        
        assert isinstance(predictions, dict)
        assert 'predictions' in predictions
        assert 'timestamp' in predictions
        assert 'metrics' in predictions
        assert len(predictions['predictions']) == len(test_features)
    
    def test_generate_predictions_missing_features(self, sample_model_results):
        """Test prediction generation with missing features."""
        selector = ModelSelector()
        best_model = selector.select_best_model(sample_model_results)
        
        # Create DataFrame with missing features
        invalid_features = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required features"):
            selector.generate_predictions(best_model, invalid_features)