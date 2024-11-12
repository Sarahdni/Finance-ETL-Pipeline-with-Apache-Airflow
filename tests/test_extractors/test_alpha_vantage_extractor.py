import pytest
import requests
from unittest.mock import patch, Mock
from src.extractors.alpha_vantage import AlphaVantageExtractor
import os
import json
import pandas as pd

class TestAlphaVantageExtractor:
    @pytest.fixture
    def extractor(self):
        """Create a test instance of AlphaVantageExtractor"""
        return AlphaVantageExtractor(api_key="test_key")

    @pytest.fixture
    def sample_response(self):
        """
        Sample API response data based on actual Alpha Vantage response
        Data format matches what we received in our manual test
        """
        return {
            "Meta Data": {
                "1. Information": "Daily Time Series with Splits and Dividend Events",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2024-11-11",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": {
                "2024-11-11": {
                    "1. open": "225.0000",
                    "2. high": "225.7000",
                    "3. low": "221.5000",
                    "4. close": "224.2300",
                    "5. volume": "42005602"
                },
                "2024-11-08": {
                    "1. open": "227.1700",
                    "2. high": "228.6600",
                    "3. low": "226.4050",
                    "4. close": "226.9600",
                    "5. volume": "38328824"
                },
                "2024-11-07": {
                    "1. open": "224.6250",
                    "2. high": "227.8750",
                    "3. low": "224.5700",
                    "4. close": "227.4800",
                    "5. volume": "42137691"
                }
            }
        }

    def test_init(self, extractor):
        """Test extractor initialization"""
        assert extractor.api_key == "test_key"
        assert extractor.base_url == "https://www.alphavantage.co/query"

    def test_init_without_api_key(self):
        """Test initialization without API key raises error"""
        with pytest.raises(ValueError):
            AlphaVantageExtractor(api_key=None)

    @patch('requests.get')
    def test_extract_daily_data_success(self, mock_get, extractor, sample_response):
        """Test successful data extraction"""
        # Configure mock
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: sample_response
        )

        # Execute test
        data = extractor.extract_daily_data("AAPL")

        # Verify the result
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'close' in data.columns
        assert len(data) == 3
        assert all(col in data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume'])
        
        # Verify the data values match our sample
        assert data.iloc[0]['close'] == 224.23
        assert data.iloc[0]['volume'] == 42005602

    @patch('requests.get')
    def test_extract_daily_data_error_response(self, mock_get, extractor):
        """Test handling of API error response"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"Error Message": "Invalid API call"}
        )

        with pytest.raises(ValueError, match="API Error"):
            extractor.extract_daily_data("AAPL")

    @patch('requests.get')
    def test_rate_limit_handling(self, mock_get, extractor):
        """Test rate limit handling"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."}
        )

        with pytest.raises(ValueError, match="Rate limit"):
            extractor.extract_daily_data("AAPL")

    def test_validate_symbol(self, extractor):
        """Test symbol validation"""
        # Valid symbols
        assert extractor._validate_symbol("AAPL") == "AAPL"
        assert extractor._validate_symbol("GOOGL") == "GOOGL"
        
        # Invalid symbols
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            extractor._validate_symbol("")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            extractor._validate_symbol("123")
        with pytest.raises(ValueError, match="Invalid symbol format"):
            extractor._validate_symbol("A" * 10)  # Too long

    def test_http_error_handling(self, extractor):
        """Test handling of HTTP errors"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("HTTP Error")
            with pytest.raises(requests.RequestException, match="HTTP Error"):
                extractor.extract_daily_data("AAPL")