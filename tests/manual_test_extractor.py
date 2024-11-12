"""
Simple script to manually test the AlphaVantage extractor.
Run this with your API key to verify the implementation works.
"""
import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.extractors.alpha_vantage import AlphaVantageExtractor
from src.config.settings import TARGET_STOCKS

def save_test_data(df: pd.DataFrame, symbol: str):
    """Save test data to CSV file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_data_{symbol}_{timestamp}.csv"
    filepath = os.path.join(project_root, 'tests', 'test_data', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath

def test_extractor():
    """Test the AlphaVantage extractor with real API calls"""
    # Load environment variables
    load_dotenv()
    
    # Verify API key
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("No API key found in .env file!")
        return
    
    logger.info("Starting extraction test...")
    logger.info(f"Testing with stocks: {TARGET_STOCKS}")
    
    # Create extractor instance
    extractor = AlphaVantageExtractor()
    
    # Test with first stock in list
    test_symbol = TARGET_STOCKS[0]
    
    try:
        logger.info(f"Testing extraction for {test_symbol}...")
        df = extractor.extract_daily_data(test_symbol)
        
        logger.info("\n=== Test Results ===")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info("\nFirst few rows:")
        print(df.head())
        
        # Save test data
        filepath = save_test_data(df, test_symbol)
        logger.info(f"\nTest data saved to: {filepath}")
        
        logger.info("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_extractor()