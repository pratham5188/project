#!/bin/bash

echo "Starting StockTrendAI Application..."
echo "=================================="

# Set up environment
source $HOME/.local/bin/env

# Run the Streamlit app
echo "Starting Streamlit server..."
uv run streamlit run app.py --server.headless true --server.port 8501

echo "Application started successfully!"
echo "Access the app at: http://localhost:8501"