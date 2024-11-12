import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query'

# Data Storage Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

# Stocks Configuration
DEFAULT_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
TARGET_STOCKS = os.getenv('TARGET_STOCKS', ','.join(DEFAULT_STOCKS)).split(',')

# Create necessary directories
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_PATH]:
    os.makedirs(path, exist_ok=True)