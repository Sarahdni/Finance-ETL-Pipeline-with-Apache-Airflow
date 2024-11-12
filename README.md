# Finance ETL Pipeline with Apache Airflow

## Overview
This project implements an automated ETL (Extract, Transform, Load) pipeline using Apache Airflow to collect, process, and analyze financial market data. The pipeline fetches stock data from Alpha Vantage API, performs technical analysis, and uses machine learning models to predict market trends.

## Features
- Automated data collection from Alpha Vantage API
- Technical indicator calculation (SMA, EMA, RSI, MACD)
- Machine learning models for price prediction
- Automated model training and selection
- Performance monitoring and logging
- Docker containerization
- Comprehensive testing suite

## Project Structure
```
finance-etl-pipeline/
├── .env.example              # Template for environment variables
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── src/                      # Source code
│   ├── __init__.py
│   ├── extractors/          # Data extraction modules
│   ├── transformers/        # Data transformation modules
│   └── ml/                  # Machine learning modules
├── dags/                    # Airflow DAG definitions
│   └── finance_etl.py       # Main ETL pipeline
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_extractors/
│   ├── test_transformers/
│   └── test_ml/
├── docs/                    # Additional documentation
└── docker/                  # Docker configuration files
```

## Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Alpha Vantage API key

## Installation
1. Clone the repository
```bash
git clone https://github.com/Saradni/finance-etl-pipeline-airflow.git
cd finance-etl-pipeline-airflow
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your Alpha Vantage API key and other configurations
```

5. Verify installation
```bash
python -m pytest tests/ -v
```

6. Start Docker containers
```bash
docker-compose up -d
```

## Development
This project is under active development. Current development phase: Initial Setup

### Current Focus
- Setting up project structure
- Implementing basic data extraction
- Establishing testing framework

## Testing
To run tests:
```bash
pytest tests/
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.