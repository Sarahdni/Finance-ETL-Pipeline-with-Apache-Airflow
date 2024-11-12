import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.transformers.data_transformer import DataTransformer

class TestDataTransformer:
    @pytest.fixture
    def sample_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 50),
            'high': np.random.uniform(150, 250, 50),
            'low': np.random.uniform(90, 180, 50),
            'close': np.random.uniform(100, 200, 50),
            'volume': np.random.uniform(1000000, 5000000, 50)
        })
        return data

    @pytest.fixture
    def transformer(self):
        """Create a DataTransformer instance"""
        return DataTransformer()

    def test_init(self, transformer):
        """Test transformer initialization"""
        assert hasattr(transformer, 'required_columns')
        assert all(col in transformer.required_columns 
                  for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_validate_data(self, transformer, sample_data):
        """Test data validation"""
        # Valid data should pass
        transformer._validate_data(sample_data)

        # Test missing columns
        invalid_data = sample_data.drop('close', axis=1)
        with pytest.raises(ValueError, match="Missing required columns"):
            transformer._validate_data(invalid_data)

        # Test empty DataFrame
        with pytest.raises(ValueError, match="Empty DataFrame"):
            transformer._validate_data(pd.DataFrame())

        # Test invalid data types
        invalid_types = sample_data.copy()
        invalid_types['close'] = 'invalid'
        with pytest.raises(ValueError, match="Invalid data type"):
            transformer._validate_data(invalid_types)

    def test_calculate_sma(self, transformer, sample_data):
        """Test Simple Moving Average calculation"""
        df = transformer._calculate_sma(sample_data)
        
        # Check if SMA columns were created
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns
        assert 'sma_200' in df.columns

        # Verify calculations
        assert df['sma_20'].equals(df['close'].rolling(window=20).mean())

    def test_calculate_ema(self, transformer, sample_data):
        """Test Exponential Moving Average calculation"""
        df = transformer._calculate_ema(sample_data)
        
        # Check if EMA columns were created
        assert 'ema_12' in df.columns
        assert 'ema_26' in df.columns

        # Verify calculations
        assert df['ema_12'].equals(df['close'].ewm(span=12, adjust=False).mean())

    def test_calculate_rsi(self, transformer, sample_data):
        """Test Relative Strength Index calculation"""
        df = transformer._calculate_rsi(sample_data)
        
        # Check if RSI column was created
        assert 'rsi' in df.columns
        
        # Verify RSI is within bounds
        assert df['rsi'].min() >= 0
        assert df['rsi'].max() <= 100

    def test_calculate_macd(self, transformer, sample_data):
        """Test MACD calculation"""
        df = transformer._calculate_macd(sample_data)
        
        # Check if MACD columns were created
        assert all(col in df.columns for col in ['macd', 'macd_signal', 'macd_hist'])

    def test_prepare_features(self, transformer, sample_data):
        """Test feature preparation"""
        features = transformer.prepare_features(sample_data)
        
        # Check if all technical indicators are present
        expected_features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26',
            'rsi',
            'macd', 'macd_signal', 'macd_hist'
        ]
        
        assert all(feature in features.columns for feature in expected_features)
        
        # Check for NaN values
        assert not features.isnull().any().any(), "Features contain NaN values"

    def test_prepare_target(self, transformer, sample_data):
        """Test target preparation"""
        target = transformer.prepare_target(sample_data)
        
        # Check target shape and type
        assert len(target) == len(sample_data) - 1  # One less due to shift
        assert isinstance(target, pd.Series)
        
        # Verify target calculation (next day's close price)
        expected_target = sample_data['close'].shift(-1).dropna()
        assert target.equals(expected_target)

    def test_transform_data(self, transformer, sample_data):
        """Test complete data transformation"""
        X, y = transformer.transform_data(sample_data)
        
        # Check shapes
        assert len(X) == len(y)
        assert not X.isnull().any().any(), "Features contain NaN values"
        assert not y.isnull().any(), "Target contains NaN values"