"""
Simple script to manually test the AlphaVantage extractor.
Run this with your API key to verify the implementation works.
"""
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.alpha_vantage import AlphaVantageExtractor

def test_extractor():
    # Load environment variables
    load_dotenv()
    
    # Create extractor instance
    extractor = AlphaVantageExtractor()
    
    # Test with a single symbol
    try:
        df = extractor.extract_daily_data("AAPL")
        print("\nSuccess! Data shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    test_extractor()