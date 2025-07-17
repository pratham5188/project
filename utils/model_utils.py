import pickle
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json

class ModelUtils:
    """Utility functions for model management and evaluation"""
    
    def __init__(self):
        self.models_dir = 'models'
        self.ensure_models_directory()
    
    def ensure_models_directory(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def save_model_metadata(self, model_name, metadata):
        """Save model metadata to JSON file"""
        try:
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            metadata['last_updated'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving metadata for {model_name}: {str(e)}")
            return False
    
    def load_model_metadata(self, model_name):
        """Load model metadata from JSON file"""
        try:
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            
            return None
        except Exception as e:
            print(f"Error loading metadata for {model_name}: {str(e)}")
            return None
    
    def evaluate_predictions(self, actual_prices, predicted_prices):
        """Evaluate model predictions using various metrics"""
        try:
            actual = np.array(actual_prices)
            predicted = np.array(predicted_prices)
            
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual = actual[mask]
            predicted = predicted[mask]
            
            if len(actual) == 0:
                return None
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Direction accuracy (for classification)
            actual_direction = np.diff(actual) > 0
            predicted_direction = np.diff(predicted) > 0
            
            if len(actual_direction) > 0:
                direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                direction_accuracy = 0
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'sample_size': len(actual)
            }
            
        except Exception as e:
            print(f"Error evaluating predictions: {str(e)}")
            return None
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.06):
        """Calculate Sharpe ratio for returns"""
        try:
            returns = np.array(returns)
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            
            if np.std(excess_returns) == 0:
                return 0
            
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe_ratio
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {str(e)}")
            return 0
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        try:
            prices = np.array(prices)
            cumulative_returns = prices / prices[0]
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            return abs(max_drawdown) * 100  # Return as percentage
            
        except Exception as e:
            print(f"Error calculating max drawdown: {str(e)}")
            return 0
    
    def generate_prediction_confidence(self, historical_accuracy, volatility, trend_strength):
        """Generate confidence score for predictions"""
        try:
            # Base confidence from historical accuracy
            base_confidence = historical_accuracy
            
            # Adjust for volatility (higher volatility = lower confidence)
            volatility_adjustment = max(0, 20 - volatility * 1000)
            
            # Adjust for trend strength (stronger trend = higher confidence)
            trend_adjustment = min(20, trend_strength * 10)
            
            # Combine factors
            confidence = base_confidence + volatility_adjustment + trend_adjustment
            
            # Ensure confidence is between 0 and 100
            confidence = max(0, min(100, confidence))
            
            return confidence
            
        except Exception as e:
            print(f"Error generating confidence score: {str(e)}")
            return 50.0  # Default confidence
    
    def backtest_strategy(self, data, predictions, initial_capital=100000):
        """Simple backtesting for prediction strategy"""
        try:
            capital = initial_capital
            positions = []
            trades = []
            
            for i in range(1, len(data)):
                if i >= len(predictions):
                    break
                
                current_price = data['Close'].iloc[i]
                previous_price = data['Close'].iloc[i-1]
                prediction = predictions[i-1]  # Previous prediction for current day
                
                # Simple strategy: buy if prediction is UP, sell if DOWN
                if prediction.get('direction') == 'UP' and len(positions) == 0:
                    # Buy
                    shares = capital // current_price
                    if shares > 0:
                        positions.append({
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares,
                            'date': data.index[i]
                        })
                        capital -= shares * current_price
                        trades.append('BUY')
                    else:
                        trades.append('HOLD')
                        
                elif prediction.get('direction') == 'DOWN' and len(positions) > 0:
                    # Sell all positions
                    for position in positions:
                        capital += position['shares'] * current_price
                    positions = []
                    trades.append('SELL')
                else:
                    trades.append('HOLD')
            
            # Calculate final portfolio value
            final_value = capital
            for position in positions:
                final_value += position['shares'] * data['Close'].iloc[-1]
            
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            return {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'number_of_trades': len([t for t in trades if t in ['BUY', 'SELL']]),
                'trades': trades
            }
            
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            return None
    
    def clean_old_models(self, max_age_days=30):
        """Clean old model files"""
        try:
            current_time = datetime.now()
            cleaned_files = []
            
            for filename in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, filename)
                
                if os.path.isfile(file_path):
                    file_age = current_time - datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_age.days > max_age_days:
                        os.remove(file_path)
                        cleaned_files.append(filename)
            
            return cleaned_files
            
        except Exception as e:
            print(f"Error cleaning old models: {str(e)}")
            return []
    
    def get_model_performance_summary(self, model_name):
        """Get comprehensive model performance summary"""
        try:
            metadata = self.load_model_metadata(model_name)
            
            if metadata is None:
                return None
            
            summary = {
                'model_name': model_name,
                'last_updated': metadata.get('last_updated', 'Unknown'),
                'accuracy': metadata.get('accuracy', 0),
                'total_predictions': metadata.get('total_predictions', 0),
                'correct_predictions': metadata.get('correct_predictions', 0),
                'avg_confidence': metadata.get('avg_confidence', 0),
                'performance_metrics': metadata.get('performance_metrics', {})
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting performance summary for {model_name}: {str(e)}")
            return None
    
    def compare_models(self, model_names):
        """Compare performance of multiple models"""
        try:
            comparison = {}
            
            for model_name in model_names:
                summary = self.get_model_performance_summary(model_name)
                if summary:
                    comparison[model_name] = summary
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
            return {}
