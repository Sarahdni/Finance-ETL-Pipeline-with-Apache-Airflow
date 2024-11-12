import os
import requests
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class AlphaVantageExtractor:
    """
    Class for extracting financial data from Alpha Vantage API.
    
    This class handles data extraction from Alpha Vantage, including rate limiting,
    error handling, and data validation.
    
    Attributes:
        api_key (str): Alpha Vantage API key
        base_url (str): Base URL for Alpha Vantage API
        calls (int): Counter for API calls
        last_call (float): Timestamp of last API call
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AlphaVantageExtractor.
        
        Args:
            api_key (str, optional): Alpha Vantage API key. If not provided, 
                                   will try to get from environment variable.
        
        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.calls = 0
        self.last_call = None

    def _validate_symbol(self, symbol: str) -> str:
        """
        Validate the stock symbol format.
        
        Args:
            symbol (str): Stock symbol to validate
            
        Returns:
            str: Validated symbol
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper()
        if not symbol.isalpha() or len(symbol) > 5:
            raise ValueError("Invalid symbol format")
            
        return symbol

    def _manage_rate_limit(self) -> None:
        """
        Manage API rate limiting.
        
        Implements a simple rate limiting mechanism to prevent exceeding
        API quotas (5 calls per minute).
        """
        if self.last_call:
            elapsed = time.time() - self.last_call
            if elapsed < 12:  # Ensure minimum 12 seconds between calls
                time.sleep(12 - elapsed)
        self.last_call = time.time()
        self.calls += 1

    def extract_daily_data(self, symbol: str) -> pd.DataFrame:
        """
        Extract daily stock data for a given symbol.
        
        Args:
            symbol (str): Stock symbol to extract data for
            
        Returns:
            pd.DataFrame: DataFrame containing the stock data
            
        Raises:
            ValueError: If API returns an error or rate limit is exceeded
            requests.RequestException: If API request fails
        """
        symbol = self._validate_symbol(symbol)
        self._manage_rate_limit()

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact"  # Last 100 data points
        }

        try:
            logger.info(f"Requesting data for {symbol}")
            response = requests.get(self.base_url, params=params, timeout=10)  # Added timeout
            response.raise_for_status()
            data = response.json()

            # Check for API errors first
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if "Note" in data and "rate limit" in data["Note"].lower():
                raise ValueError(f"Rate limit exceeded: {data['Note']}")

            # Extract time series data
            if "Time Series (Daily)" not in data:
                raise ValueError("Invalid API response format")

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            
            # Clean column names
            df.columns = [col.split(". ")[1] for col in df.columns]
            
            # Convert values to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Add date as column
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)

            logger.info(f"Successfully extracted {len(df)} records for {symbol}")
            return df

        except requests.RequestException as e:
            logger.error(f"Request failed for {symbol}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting data for {symbol}: {str(e)}")
            raise