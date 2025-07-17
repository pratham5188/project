#!/usr/bin/env python3
"""
Test script to validate all new AI models for StockTrendAI
"""

import pandas as pd
import numpy as np
import sys
import traceback
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock price data
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, len(dates))
    })
    
    data.set_index('Date', inplace=True)
    
    # Add technical indicators
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['RSI'] = 50 + np.random.normal(0, 10, len(data))  # Simplified RSI
    data['MACD'] = np.random.normal(0, 1, len(data))
    data['MACD_Signal'] = data['MACD'].rolling(9).mean()
    data['BB_Upper'] = data['Close'] * 1.02
    data['BB_Lower'] = data['Close'] * 0.98
    data['Volatility'] = data['Close'].rolling(20).std()
    
    return data.dropna()

def test_model_import(model_name):
    """Test if a model can be imported"""
    try:
        if model_name == 'XGBoost':
            from models.xgboost_model import XGBoostPredictor
            return XGBoostPredictor(), True
        elif model_name == 'LSTM':
            from models.lstm_model import LSTMPredictor
            return LSTMPredictor(), True
        elif model_name == 'Prophet':
            from models.prophet_model import ProphetPredictor
            return ProphetPredictor(), True
        elif model_name == 'GRU':
            from models.gru_model import GRUPredictor
            return GRUPredictor(), True
        elif model_name == 'Transformer':
            from models.transformer_model import TransformerPredictor
            return TransformerPredictor(), True
        elif model_name == 'Ensemble':
            from models.ensemble_model import EnsemblePredictor
            return EnsemblePredictor(), True
        else:
            return None, False
    except Exception as e:
        print(f"❌ Failed to import {model_name}: {str(e)}")
        traceback.print_exc()
        return None, False

def test_model_prediction(model, model_name, data):
    """Test if a model can make predictions"""
    try:
        prediction = model.predict(data)
        if prediction and isinstance(prediction, dict):
            required_keys = ['direction', 'confidence', 'predicted_price']
            if all(key in prediction for key in required_keys):
                return True, prediction
            else:
                return False, f"Missing required keys: {[k for k in required_keys if k not in prediction]}"
        else:
            return False, "Invalid prediction format"
    except Exception as e:
        return False, str(e)

def main():
    """Main test function"""
    print("🚀 Testing StockTrendAI Advanced Models")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    try:
        data = create_sample_data()
        print(f"✅ Sample data created: {len(data)} rows")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Columns: {list(data.columns)}")
    except Exception as e:
        print(f"❌ Failed to create sample data: {str(e)}")
        return
    
    print("\n" + "=" * 50)
    print("🤖 Testing Model Imports and Predictions")
    print("=" * 50)
    
    models_to_test = ['XGBoost', 'LSTM', 'Prophet', 'GRU', 'Transformer', 'Ensemble']
    results = {}
    
    for model_name in models_to_test:
        print(f"\n📋 Testing {model_name} Model...")
        
        # Test import
        model, import_success = test_model_import(model_name)
        if not import_success:
            results[model_name] = {'import': False, 'predict': False}
            continue
        
        print(f"   ✅ Import successful")
        
        # Test prediction
        predict_success, prediction_result = test_model_prediction(model, model_name, data)
        
        if predict_success:
            print(f"   ✅ Prediction successful")
            print(f"      Direction: {prediction_result['direction']}")
            print(f"      Confidence: {prediction_result['confidence']:.1f}%")
            print(f"      Predicted Price: ${prediction_result['predicted_price']:.2f}")
            print(f"      Model Type: {prediction_result.get('model_type', 'Unknown')}")
            results[model_name] = {'import': True, 'predict': True, 'result': prediction_result}
        else:
            print(f"   ❌ Prediction failed: {prediction_result}")
            results[model_name] = {'import': True, 'predict': False, 'error': prediction_result}
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 Test Results Summary")
    print("=" * 50)
    
    successful_imports = sum(1 for r in results.values() if r['import'])
    successful_predictions = sum(1 for r in results.values() if r.get('predict', False))
    
    print(f"Total Models Tested: {len(models_to_test)}")
    print(f"Successful Imports: {successful_imports}/{len(models_to_test)}")
    print(f"Successful Predictions: {successful_predictions}/{len(models_to_test)}")
    
    print("\nDetailed Results:")
    for model_name, result in results.items():
        status = "✅" if result.get('predict', False) else "❌"
        print(f"  {status} {model_name}: Import={result['import']}, Predict={result.get('predict', False)}")
        if not result.get('predict', False) and 'error' in result:
            print(f"     Error: {result['error']}")
    
    # Test ensemble with multiple models if available
    if successful_predictions >= 2:
        print(f"\n🎯 Ensemble Model Analysis")
        print("-" * 30)
        
        ensemble_result = results.get('Ensemble', {})
        if ensemble_result.get('predict', False):
            ensemble_pred = ensemble_result['result']
            if 'base_predictions' in ensemble_pred:
                print(f"Base model predictions:")
                for base_model, base_pred in ensemble_pred['base_predictions'].items():
                    print(f"  {base_model}: {base_pred['direction']} ({base_pred['confidence']:.1f}%)")
                
                if 'weights' in ensemble_pred:
                    print(f"Model weights:")
                    for model, weight in ensemble_pred['weights'].items():
                        print(f"  {model}: {weight:.3f}")
                
                print(f"Consensus: {ensemble_pred.get('consensus', 'N/A'):.1f}")
                print(f"Meta-learner used: {ensemble_pred.get('meta_used', 'N/A')}")
    
    print(f"\n🎉 Testing completed!")
    if successful_predictions == len(models_to_test):
        print("🏆 All models are working perfectly!")
    elif successful_predictions > 0:
        print(f"⚠️  {successful_predictions} out of {len(models_to_test)} models are working.")
    else:
        print("❌ No models are working. Please check dependencies and configurations.")

if __name__ == "__main__":
    main()