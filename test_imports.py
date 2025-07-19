#!/usr/bin/env python3
"""
Test script to verify all imports work correctly for deployment
"""

import sys
print(f"Python version: {sys.version}")

# Test basic imports
try:
    import streamlit
    print("✓ Streamlit imported successfully")
except ImportError as e:
    print(f"✗ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import plotly
    print("✓ Plotly imported successfully")
except ImportError as e:
    print(f"✗ Plotly import failed: {e}")

try:
    import sklearn
    print("✓ Scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import yfinance
    print("✓ YFinance imported successfully")
except ImportError as e:
    print(f"✗ YFinance import failed: {e}")

try:
    import xgboost
    print("✓ XGBoost imported successfully")
except ImportError as e:
    print(f"✗ XGBoost import failed: {e}")

try:
    import textblob
    print("✓ TextBlob imported successfully")
except ImportError as e:
    print(f"✗ TextBlob import failed: {e}")

try:
    import prophet
    print("✓ Prophet imported successfully")
except ImportError as e:
    print(f"✗ Prophet import failed: {e}")

try:
    import lightgbm
    print("✓ LightGBM imported successfully")
except ImportError as e:
    print(f"✗ LightGBM import failed: {e}")

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")

try:
    import keras
    print("✓ Keras imported successfully")
except ImportError as e:
    print(f"✗ Keras import failed: {e}")

try:
    import jax
    print("✓ JAX imported successfully")
except ImportError as e:
    print(f"✗ JAX import failed: {e}")

try:
    import jaxlib
    print("✓ JAXlib imported successfully")
except ImportError as e:
    print(f"✗ JAXlib import failed: {e}")

# Test app imports
try:
    from utils.data_fetcher import DataFetcher
    print("✓ DataFetcher imported successfully")
except ImportError as e:
    print(f"✗ DataFetcher import failed: {e}")

try:
    from utils.technical_indicators import TechnicalIndicators
    print("✓ TechnicalIndicators imported successfully")
except ImportError as e:
    print(f"✗ TechnicalIndicators import failed: {e}")

try:
    from models.xgboost_model import XGBoostPredictor
    print("✓ XGBoostPredictor imported successfully")
except ImportError as e:
    print(f"✗ XGBoostPredictor import failed: {e}")

try:
    from config.settings import DEFAULT_STOCK
    print("✓ Settings imported successfully")
except ImportError as e:
    print(f"✗ Settings import failed: {e}")

try:
    from styles.custom_css import get_custom_css
    print("✓ Custom CSS imported successfully")
except ImportError as e:
    print(f"✗ Custom CSS import failed: {e}")

print("\n=== Import Test Complete ===")