#!/bin/bash

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export STREAMLIT_SERVER_PORT=5000
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install dependencies if needed
echo "Checking dependencies..."
pip3 install --break-system-packages -r requirements.txt

# Run the Streamlit app
echo "Starting StockTrendAI application..."
python3 -m streamlit run app.py --server.port 5000 --server.headless true