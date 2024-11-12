"""
ETL DAG for financial analysis.

This DAG performs the following operations:
1. Stock data extraction via Alpha Vantage API
2. Data transformation and technical indicators calculation
3. ML model training and selection
4. Prediction generation

Author: Sarah DNI
Date: 2024-03-12
"""

import os
from dotenv import load_dotenv
import pandas as pd
import time
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

from src.extractors.alpha_vantage import AlphaVantageExtractor
from src.transformers.data_transformer import DataTransformer
from src.ml.models import ModelTrainer, ModelSelector

# Configuration
load_dotenv()

TARGET_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def extract_stock_data(**context) -> None:
    """
    Extracts stock data from Alpha Vantage API.
    
    This function:
    1. Connects to Alpha Vantage API
    2. Retrieves daily data for each symbol
    3. Stores raw data in XCom
    
    Args:
        **context: Airflow context for XCom access
        
    Raises:
        Exception: If any error occurs during data extraction
    """
    try:
        extractor = AlphaVantageExtractor()
        all_stocks_data = {}
        
        logger.info(f"Starting extraction for symbols: {TARGET_STOCKS}")
        
        for symbol in TARGET_STOCKS:
            try:
                # Wait 12 seconds between calls (5 calls/minute max)
                if symbol != TARGET_STOCKS[0]:
                    logger.info(f"Waiting 12 seconds to respect API rate limit for {symbol}...")
                    time.sleep(12)
                
                data = extractor.extract_daily_data(symbol)
                if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                    all_stocks_data[symbol] = data
                    logger.info(f"Data successfully extracted for {symbol}, shape: {data.shape}")
                else:
                    logger.warning(f"Invalid or empty data received for {symbol}")
            except Exception as e:
                logger.error(f"Error extracting data for {symbol}: {str(e)}")
                continue
        
        if not all_stocks_data:
            raise ValueError("No data successfully extracted for any symbol")
        
        logger.info(f"Data successfully extracted for {len(all_stocks_data)} symbols")
        context['task_instance'].xcom_push(key='raw_stock_data', value=all_stocks_data)
    except Exception as e:
        logger.error(f"Failed to extract stock data: {str(e)}")
        raise

def process_data(**context) -> None:
    """
    Processes raw stock data and calculates technical indicators.
    
    This function:
    1. Retrieves raw data from XCom
    2. Validates data structure
    3. Calculates technical indicators
    4. Prepares features for ML
    5. Stores results in XCom
    
    Args:
        **context: Airflow context for XCom access
        
    Raises:
        ValueError: If no valid data is available for processing
        Exception: For any other processing error
    """
    try:
        raw_data = context['task_instance'].xcom_pull(key='raw_stock_data')
        if not raw_data:
            raise ValueError("No raw data available in XCom")
            
        logger.info(f"Processing data for {len(raw_data)} symbols")
        
        transformer = DataTransformer()
        processed_data = {}
        
        for symbol, data in raw_data.items():
            try:
                if not isinstance(data, pd.DataFrame):
                    logger.error(f"Invalid data type for {symbol}: {type(data)}")
                    continue
                
                processed_df = transformer.transform_data(data)
                processed_data[symbol] = processed_df
                logger.info(f"Successfully processed data for {symbol}")
            except Exception as e:
                logger.error(f"Error processing data for {symbol}: {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("No data successfully processed")
        
        context['task_instance'].xcom_push(key='processed_data', value=processed_data)
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")
        raise

def train_evaluate_models(**context) -> None:
    """
    Trains and evaluates ML models on the processed data.
    
    This function:
    1. Retrieves processed data from XCom
    2. Splits data into training and testing sets
    3. Trains multiple models
    4. Evaluates their performance
    5. Stores results in XCom
    
    Args:
        **context: Airflow context for XCom access
        
    Raises:
        ValueError: If training data is invalid
        Exception: For any other training error
    """
    try:
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        if not processed_data:
            raise ValueError("No processed data available in XCom")
        
        trainer = ModelTrainer()
        model_results = {}
        
        for symbol, data in processed_data.items():
            try:
                model_result = trainer.train_models(data)
                model_results[symbol] = model_result
                logger.info(f"Models trained successfully for {symbol}")
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {str(e)}")
                continue
        
        if not model_results:
            raise ValueError("No models successfully trained")
        
        context['task_instance'].xcom_push(key='model_results', value=model_results)
        logger.info("Model training and evaluation completed successfully")
    except Exception as e:
        logger.error(f"Failed to train and evaluate models: {str(e)}")
        raise

def generate_predictions(**context) -> None:
    """
    Generates predictions using the trained models.
    
    This function:
    1. Retrieves model results and processed data from XCom
    2. Selects best model for each symbol
    3. Generates predictions
    4. Stores results in XCom
    
    Args:
        **context: Airflow context for XCom access
        
    Raises:
        Exception: If any error occurs during prediction generation
    """
    try:
        model_results = context['task_instance'].xcom_pull(key='model_results')
        processed_data = context['task_instance'].xcom_pull(key='processed_data')
        
        if not model_results or not processed_data:
            raise ValueError("Missing required data in XCom")
        
        predictions = {}
        for symbol in processed_data.keys():
            try:
                model_selector = ModelSelector()
                symbol_predictions = model_selector.generate_predictions(
                    model_results[symbol],
                    processed_data[symbol]
                )
                predictions[symbol] = symbol_predictions
                logger.info(f"Predictions generated successfully for {symbol}")
            except Exception as e:
                logger.error(f"Error generating predictions for {symbol}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No predictions generated for any symbol")
        
        context['task_instance'].xcom_push(key='predictions', value=predictions)
        logger.info("Prediction generation completed successfully")
    except Exception as e:
        logger.error(f"Failed to generate predictions: {str(e)}")
        raise

# DAG creation
with DAG(
    'finance_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for financial analysis',
    schedule='@daily',  # Using schedule instead of schedule_interval
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    # Data extraction task
    extract_data = PythonOperator(
        task_id='extract_stock_data',
        python_callable=extract_stock_data,
        dag=dag
    )

    # Data processing task
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        dag=dag
    )

    # Machine learning task group
    with TaskGroup('machine_learning') as ml_group:
        # Model training
        train_models = PythonOperator(
            task_id='train_evaluate_models',
            python_callable=train_evaluate_models
        )

        # Prediction generation
        make_predictions = PythonOperator(
            task_id='generate_predictions',
            python_callable=generate_predictions
        )

        # ML group dependencies
        train_models >> make_predictions

    # Main dependencies
    extract_data >> process_data_task >> ml_group