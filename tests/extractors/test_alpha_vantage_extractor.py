import pytest
import time
from unittest.mock import patch
from dotenv import load_dotenv
from src.extractors.alpha_vantage import AlphaVantageExtractor

# Load environment variables at the module level
load_dotenv()

class TestAlphaVantageExtractor:
    """Test suite for AlphaVantageExtractor class."""

    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        extractor = AlphaVantageExtractor(api_key="test_key")
        assert extractor.api_key == "test_key"
        assert extractor.base_url == "https://www.alphavantage.co/query"
        assert extractor.calls == 0
        assert extractor.last_call is None

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):  # Clear environment variables
            with pytest.raises(ValueError, match="Alpha Vantage API key is required"):
                AlphaVantageExtractor()

    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict('os.environ', {'ALPHA_VANTAGE_API_KEY': 'env_test_key'}):
            extractor = AlphaVantageExtractor()
            assert extractor.api_key == "env_test_key"

    def test_validate_symbol_valid(self):
        """Test symbol validation with valid input."""
        extractor = AlphaVantageExtractor(api_key="test_key")
        assert extractor._validate_symbol("AAPL") == "AAPL"
        assert extractor._validate_symbol("aapl") == "AAPL"

    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid input."""
        extractor = AlphaVantageExtractor(api_key="test_key")
        invalid_symbols = ["", None, "A" * 6, "123", "A@PL"]
        for symbol in invalid_symbols:
            with pytest.raises(ValueError):
                extractor._validate_symbol(symbol)

    def test_manage_rate_limit(self):
        """Test rate limiting functionality."""
        extractor = AlphaVantageExtractor(api_key="test_key")
        start_time = time.time()
        extractor._manage_rate_limit()
        extractor._manage_rate_limit()
        end_time = time.time()
        assert end_time - start_time >= 12  # Vérifier le délai minimal entre les appels

    @patch('requests.get')
    def test_extract_daily_data_success(self, mock_get):
        """Test successful data extraction."""
        mock_response = {
            "Time Series (Daily)": {
                "2024-03-12": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.0",
                    "4. close": "100.5",
                    "5. volume": "1000000"
                }
            }
        }
        mock_get.return_value.json.return_value = mock_response
        mock_get.return_value.status_code = 200

        extractor = AlphaVantageExtractor(api_key="test_key")
        df = extractor.extract_daily_data("AAPL")
        
        assert not df.empty
        assert "close" in df.columns
        assert len(df) > 0