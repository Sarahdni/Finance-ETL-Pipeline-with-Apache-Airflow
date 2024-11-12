"""
Machine Learning models for stock price prediction.

This module implements:
1. ModelTrainer: Trains and evaluates multiple ML models
2. ModelSelector: Selects best model and generates predictions

"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models for stock prediction.
    
    This class handles:
    - Data preparation
    - Model training
    - Model evaluation
    - Performance metrics calculation
    """
    
    def __init__(self):
        """Initialize model configurations and evaluation metrics."""
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.scaler = StandardScaler()
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple containing features array and target array
            
        Raises:
            ValueError: If data validation fails
        """
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        # Separate features and target
        X = data.drop(['target'], axis=1)
        y = data['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y.values
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = model.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Dictionary with model results and metrics
            
        Raises:
            ValueError: If data preparation fails
        """
        try:
            logger.info("Starting model training process")
            
            # Prepare data
            X, y = self.prepare_data(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results = {}
            
            # Train and evaluate each model
            for name, model in self.models.items():
                try:
                    logger.info(f"Training {name} model")
                    model.fit(X_train, y_train)
                    
                    # Evaluate performance
                    metrics = self.evaluate_model(model, X_test, y_test)
                    
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'scaler': self.scaler,
                        'feature_names': data.drop(['target'], axis=1).columns.tolist()
                    }
                    
                    logger.info(f"{name} model trained successfully. RMSE: {metrics['rmse']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {name} model: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No models were successfully trained")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}")
            raise

class ModelSelector:
    """
    Selects best performing model and generates predictions.
    
    This class handles:
    - Model selection based on metrics
    - Prediction generation
    - Result formatting
    """
    
    def select_best_model(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select best model based on RMSE.
        
        Args:
            model_results: Dictionary of model results and metrics
            
        Returns:
            Dictionary containing best model info
            
        Raises:
            ValueError: If no valid models are provided
        """
        if not model_results:
            raise ValueError("No model results provided")
            
        # Select based on RMSE
        best_model_name = min(
            model_results.keys(),
            key=lambda k: model_results[k]['metrics']['rmse']
        )
        
        return model_results[best_model_name]
    
    def generate_predictions(self, model_info: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions using selected model.
        
        Args:
            model_info: Dictionary with model and preprocessing info
            features: DataFrame with features for prediction
            
        Returns:
            Dictionary with predictions and confidence metrics
            
        Raises:
            ValueError: If invalid data is provided
        """
        try:
            # Validate features
            required_features = model_info['feature_names']
            if not all(feat in features.columns for feat in required_features):
                raise ValueError("Missing required features")
                
            # Prepare features
            X = features[required_features]
            X_scaled = model_info['scaler'].transform(X)
            
            # Generate predictions
            predictions = model_info['model'].predict(X_scaled)
            
            return {
                'predictions': predictions.tolist(),
                'timestamp': pd.Timestamp.now().isoformat(),
                'metrics': model_info['metrics']
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise