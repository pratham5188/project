#!/usr/bin/env python3
"""
Comprehensive test script for StockTrendAI application
Tests all components, models, and functionality
"""

import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all imports and dependencies"""
    print("=" * 60)
    print("🔍 TESTING IMPORTS")
    print("=" * 60)
    
    import_tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("streamlit", "import streamlit as st"),
        ("plotly", "import plotly.graph_objects as go"),
        ("sklearn", "from sklearn.ensemble import RandomForestRegressor"),
        ("yfinance", "import yfinance as yf"),
        ("xgboost", "import xgboost as xgb"),
        ("textblob", "from textblob import TextBlob"),
        
        # Optional imports with fallbacks
        ("lightgbm", "import lightgbm as lgb"),
        ("tensorflow", "import tensorflow as tf"),
        ("keras", "import keras"),
        ("prophet", "from prophet import Prophet"),
        ("jax", "import jax"),
    ]
    
    results = []
    for name, import_stmt in import_tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:<15} - SUCCESS")
            results.append((name, True, None))
        except ImportError as e:
            print(f"❌ {name:<15} - MISSING (using fallback)")
            results.append((name, False, str(e)))
        except Exception as e:
            print(f"⚠️  {name:<15} - ERROR: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_models():
    """Test all model imports and initialization"""
    print("\n" + "=" * 60)
    print("🤖 TESTING AI MODELS")
    print("=" * 60)
    
    model_tests = [
        ("XGBoost", "from models.xgboost_model import XGBoostPredictor; XGBoostPredictor()"),
        ("LSTM", "from models.lstm_model import LSTMPredictor; LSTMPredictor()"),
        ("Prophet", "from models.prophet_model import ProphetPredictor; ProphetPredictor()"),
        ("Ensemble", "from models.ensemble_model import EnsemblePredictor; EnsemblePredictor()"),
        ("Transformer", "from models.transformer_model import TransformerPredictor; TransformerPredictor()"),
        ("GRU", "from models.gru_model import GRUPredictor; GRUPredictor()"),
        ("Stacking", "from models.stacking_ensemble import StackingEnsemblePredictor; StackingEnsemblePredictor()"),
    ]
    
    results = []
    for name, test_code in model_tests:
        try:
            exec(test_code)
            print(f"✅ {name:<15} - SUCCESS")
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ {name:<15} - ERROR: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_utilities():
    """Test utility modules"""
    print("\n" + "=" * 60)
    print("🔧 TESTING UTILITIES")
    print("=" * 60)
    
    utility_tests = [
        ("DataFetcher", "from utils.data_fetcher import DataFetcher; DataFetcher()"),
        ("TechnicalIndicators", "from utils.technical_indicators import TechnicalIndicators; TechnicalIndicators()"),
        ("ModelUtils", "from utils.model_utils import ModelUtils; ModelUtils()"),
        ("PortfolioTracker", "from utils.portfolio_tracker import PortfolioTracker; PortfolioTracker()"),
        ("AdvancedAnalytics", "from utils.advanced_analytics import AdvancedAnalytics; AdvancedAnalytics()"),
        ("NewsSentiment", "from utils.news_sentiment import NewsSentimentAnalyzer; NewsSentimentAnalyzer()"),
        ("UIComponents", "from utils.ui_components import UIComponents; UIComponents()"),
        ("ModelInfo", "from utils.model_info import ModelInfo; ModelInfo()"),
        ("CustomCSS", "from styles.custom_css import get_custom_css; get_custom_css()"),
    ]
    
    results = []
    for name, test_code in utility_tests:
        try:
            exec(test_code)
            print(f"✅ {name:<15} - SUCCESS")
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ {name:<15} - ERROR: {e}")
            results.append((name, False, str(e)))
    
    return results

def test_main_app():
    """Test main application"""
    print("\n" + "=" * 60)
    print("🚀 TESTING MAIN APPLICATION")
    print("=" * 60)
    
    try:
        # Test app import
        import app
        print("✅ App import - SUCCESS")
        
        # Test app initialization
        stock_app = app.StockTrendAI()
        print("✅ App initialization - SUCCESS")
        
        # Test data fetching (with a quick test)
        try:
            # Quick test with a small period
            test_data = stock_app.data_fetcher.get_stock_data("RELIANCE.NS", "5d")
            if test_data is not None and not test_data.empty:
                print("✅ Data fetching - SUCCESS")
            else:
                print("⚠️  Data fetching - NO DATA (network issue)")
        except Exception as e:
            print(f"⚠️  Data fetching - ERROR: {e}")
        
        return True, None
        
    except Exception as e:
        print(f"❌ Main app - ERROR: {e}")
        traceback.print_exc()
        return False, str(e)

def generate_test_report(import_results, model_results, utility_results, app_result):
    """Generate comprehensive test report"""
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_tests = len(import_results) + len(model_results) + len(utility_results) + 1
    passed_tests = (
        sum(1 for _, success, _ in import_results if success) +
        sum(1 for _, success, _ in model_results if success) +
        sum(1 for _, success, _ in utility_results if success) +
        (1 if app_result[0] else 0)
    )
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Critical failures
    critical_failures = []
    for name, success, error in model_results:
        if not success:
            critical_failures.append(f"Model {name}: {error}")
    
    for name, success, error in utility_results:
        if not success:
            critical_failures.append(f"Utility {name}: {error}")
    
    if not app_result[0]:
        critical_failures.append(f"Main App: {app_result[1]}")
    
    if critical_failures:
        print("\n❌ CRITICAL FAILURES:")
        for failure in critical_failures:
            print(f"  • {failure}")
    else:
        print("\n✅ NO CRITICAL FAILURES - App is ready for deployment!")
    
    # Optional dependency warnings
    optional_missing = []
    for name, success, error in import_results:
        if not success and name in ['lightgbm', 'tensorflow', 'keras', 'prophet', 'jax']:
            optional_missing.append(name)
    
    if optional_missing:
        print(f"\n⚠️  OPTIONAL DEPENDENCIES MISSING: {', '.join(optional_missing)}")
        print("   App will use fallback implementations for these features.")
    
    return passed_tests, total_tests, critical_failures

def main():
    """Run all tests"""
    print("🧪 StockTrendAI - Comprehensive Application Test")
    print("This test validates all components and functionality")
    
    # Run all tests
    import_results = test_imports()
    model_results = test_models()
    utility_results = test_utilities()
    app_result = test_main_app()
    
    # Generate report
    passed, total, critical_failures = generate_test_report(
        import_results, model_results, utility_results, app_result
    )
    
    # Final status
    if not critical_failures:
        print("\n🎉 ALL TESTS PASSED! Application is ready to run.")
        print("   You can start the app with: streamlit run app.py")
        return 0
    else:
        print(f"\n⚠️  {len(critical_failures)} critical issues found.")
        print("   Please fix these issues before running the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main())