import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("Prophet library loaded successfully")
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available - using trend-based fallback")

class ProphetPredictor:
    """Facebook Prophet-based stock prediction model for time series analysis"""
    
    def __init__(self):
        self.model = None
        self.prophet_available = PROPHET_AVAILABLE
        
    def prepare_prophet_data(self, data):
        """Prepare data in Prophet format (ds, y)"""
        prophet_data = pd.DataFrame()
        prophet_data['ds'] = data.index
        prophet_data['y'] = data['Close'].values
        
        # Add additional regressors
        if 'Volume' in data.columns:
            prophet_data['volume'] = data['Volume'].values
        if 'RSI' in data.columns:
            prophet_data['rsi'] = data['RSI'].values
        if 'MACD' in data.columns:
            prophet_data['macd'] = data['MACD'].values
            
        return prophet_data
    
    def train(self, data):
        """Train Prophet model"""
        try:
            if not self.prophet_available:
                print("Prophet not available for training")
                return None
            
            # Prepare data
            prophet_data = self.prepare_prophet_data(data)
            
            if len(prophet_data) < 30:
                raise ValueError("Insufficient data for Prophet training")
            
            # Initialize Prophet with custom parameters for financial data
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,  # More flexible for volatile stocks
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                seasonality_mode='multiplicative',
                interval_width=0.8
            )
            
            # Add additional regressors if available
            if 'volume' in prophet_data.columns:
                self.model.add_regressor('volume')
            if 'rsi' in prophet_data.columns:
                self.model.add_regressor('rsi')
            if 'macd' in prophet_data.columns:
                self.model.add_regressor('macd')
            
            # Fit the model
            self.model.fit(prophet_data)
            
            return True
            
        except Exception as e:
            print(f"Prophet training error: {str(e)}")
            return None
    
    def trend_based_prediction(self, data):
        """Fallback prediction using trend analysis"""
        # Calculate multiple timeframe trends
        short_trend = data['Close'].rolling(window=5).mean().pct_change().iloc[-1]
        medium_trend = data['Close'].rolling(window=20).mean().pct_change().iloc[-1]
        long_trend = data['Close'].rolling(window=50).mean().pct_change().iloc[-1] if len(data) >= 50 else 0
        
        # Weight the trends
        weighted_trend = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
        
        current_price = data['Close'].iloc[-1]
        
        # Calculate predicted price
        if weighted_trend > 0.01:  # Strong upward trend
            direction = 'UP'
            confidence = min(75.0, 60.0 + abs(weighted_trend) * 1000)
            predicted_price = current_price * (1 + abs(weighted_trend) * 2)
        elif weighted_trend < -0.01:  # Strong downward trend
            direction = 'DOWN'
            confidence = min(75.0, 60.0 + abs(weighted_trend) * 1000)
            predicted_price = current_price * (1 - abs(weighted_trend) * 2)
        else:  # Sideways trend
            direction = 'UP' if current_price > data['Close'].iloc[-2] else 'DOWN'
            confidence = 55.0
            predicted_price = current_price * (1.005 if direction == 'UP' else 0.995)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'Prophet (Trend Fallback)'
        }
    
    def predict(self, data):
        """Make Prophet prediction for next day"""
        try:
            if not self.prophet_available:
                return self.trend_based_prediction(data)
            
            # Train model if not already trained
            if self.model is None:
                print("Training new Prophet model...")
                result = self.train(data.copy())
                if result is None:
                    return self.trend_based_prediction(data)
            
            # Create future dataframe for 1 day
            last_date = data.index[-1]
            next_date = last_date + timedelta(days=1)
            
            future = pd.DataFrame({'ds': [next_date]})
            
            # Add regressor values for prediction (use last known values)
            if 'Volume' in data.columns and hasattr(self.model, 'extra_regressors'):
                if 'volume' in [reg for reg in self.model.extra_regressors]:
                    future['volume'] = data['Volume'].iloc[-1]
            if 'RSI' in data.columns and hasattr(self.model, 'extra_regressors'):
                if 'rsi' in [reg for reg in self.model.extra_regressors]:
                    future['rsi'] = data['RSI'].iloc[-1]
            if 'MACD' in data.columns and hasattr(self.model, 'extra_regressors'):
                if 'macd' in [reg for reg in self.model.extra_regressors]:
                    future['macd'] = data['MACD'].iloc[-1]
            
            # Make prediction
            forecast = self.model.predict(future)
            predicted_price = forecast['yhat'].iloc[0]
            
            # Calculate confidence based on prediction interval
            lower_bound = forecast['yhat_lower'].iloc[0]
            upper_bound = forecast['yhat_upper'].iloc[0]
            prediction_range = upper_bound - lower_bound
            current_price = data['Close'].iloc[-1]
            
            # Calculate confidence (smaller range = higher confidence)
            confidence = max(50.0, min(85.0, 100 - (prediction_range / current_price) * 100))
            
            # Determine direction
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Ensure predicted price is reasonable
            max_change = 0.1  # 10% max change
            if abs(predicted_price - current_price) / current_price > max_change:
                if predicted_price > current_price:
                    predicted_price = current_price * (1 + max_change)
                else:
                    predicted_price = current_price * (1 - max_change)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'model_type': 'Prophet (Time Series)',
                'forecast_components': {
                    'trend': forecast.get('trend', [0]).iloc[0] if 'trend' in forecast.columns else 0,
                    'seasonal': forecast.get('seasonal', [0]).iloc[0] if 'seasonal' in forecast.columns else 0,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            }
            
        except Exception as e:
            print(f"Prophet prediction error: {str(e)}")
            return self.trend_based_prediction(data)