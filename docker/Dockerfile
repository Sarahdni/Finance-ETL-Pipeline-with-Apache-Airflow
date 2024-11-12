# Base image with Python and Airflow
FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements first for better caching
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy project files
COPY --chown=airflow:root . /opt/airflow/finance_etl

# Add project directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow/finance_etl"

# Initialize Airflow DB and create admin user
RUN airflow db init && \
    airflow users create \
        --username admin \
        --password admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com