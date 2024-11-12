#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up Finance ETL Pipeline...${NC}"

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Create necessary directories
echo "Creating required directories..."
mkdir -p ./dags ./logs ./plugins

# Copy environment variables
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please update .env with your configurations"
fi

# Build and start services
echo "Building and starting services..."
docker-compose up -d --build

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose ps

echo -e "${GREEN}Setup complete! Access Airflow at http://localhost:8080${NC}"
echo "Default credentials:"
echo "Username: admin"
echo "Password: admin"