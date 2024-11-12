import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class DataTransformer:
    """
    Transform raw stock data into features for machine learning.
    
    This class handles the calculation of technical indicators and
    preparation of features for machine learning models.
    """
    
    def __init__(self):
        """Initialize the DataTransformer with required columns"""
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate input data format and content.
        
        Args:
            df (pd.DataFrame): Input DataFrame to validate
            
        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
            
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Invalid data type for column {col}")

    def _calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Average for specified periods.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            periods (List[int]): List of periods for SMA calculation
            
        Returns:
            pd.DataFrame: DataFrame with added SMA columns
        """
        df = df.copy()
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    def _calculate_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average for specified periods.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            periods (List[int]): List of periods for EMA calculation
            
        Returns:
            pd.DataFrame: DataFrame with added EMA columns
        """
        df = df.copy()
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            period (int): Period for RSI calculation
            
        Returns:
            pd.DataFrame: DataFrame with added RSI column
        """
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with added MACD columns
        """
        df = df.copy()
        
        # Calculate EMAs first
        df = self._calculate_ema(df)
        
        # Calculate MACD line
        df['macd'] = df['ema_12'] - df['ema_26']
        
        # Calculate signal line (9-day EMA of MACD)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature set from raw data.
        
        Args:
            df (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Prepared features
        """
        try:
            self._validate_data(df)
            
            features = df.copy()
            
            # Calculate all technical indicators
            features = self._calculate_sma(features)
            features = self._calculate_ema(features)
            features = self._calculate_rsi(features)
            features = self._calculate_macd(features)
            
            # Drop NaN values created by indicators
            features = features.dropna()
            
            logger.info(f"Prepared features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Prepare target variable (next day's closing price).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.Series: Target variable
        """
        try:
            target = df['close'].shift(-1)
            target = target.dropna()
            
            logger.info(f"Prepared target shape: {target.shape}")
            return target
            
        except Exception as e:
            logger.error(f"Error preparing target: {str(e)}")
            raise

    def transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform raw data into features and target for machine learning.
        
        Args:
            df (pd.DataFrame): Raw input data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            logger.info("Starting data transformation")
            
            # Prepare features and target
            features = self.prepare_features(df)
            target = self.prepare_target(features)
            
            # Align features and target
            features = features[:-1]  # Remove last row since we don't have next day's price
            
            logger.info("Data transformation completed successfully")
            return features, target
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise