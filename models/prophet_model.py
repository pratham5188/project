import pandas as pd
import numpy as np
import warnings
import os
import pickle
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    print("Prophet library loaded successfully")
except ImportError as e:
    PROPHET_AVAILABLE = False
    print(f"Prophet not available - using fallback implementation: {e}")

class ProphetPredictor:
    """Facebook Prophet-based stock prediction model for time series forecasting"""
    
    def __init__(self):
        self.model = None
        self.model_path = 'models/prophet_model.pkl'
        self.prophet_available = PROPHET_AVAILABLE
        self.last_training_data = None
        
        if not self.prophet_available:
            print("Prophet not available, using simplified time series alternative")
    
    def prepare_prophet_data(self, data):
        """Prepare data for Prophet model (requires 'ds' and 'y' columns)"""
        prophet_data = pd.DataFrame()
        prophet_data['ds'] = data.index  # Date column
        prophet_data['y'] = data['Close'].values  # Target variable (Close price)
        
        # Add additional regressors
        if 'Volume' in data.columns:
            prophet_data['volume'] = data['Volume'].values
        if 'RSI' in data.columns:
            prophet_data['rsi'] = data['RSI'].values
        if 'MACD' in data.columns:
            prophet_data['macd'] = data['MACD'].values
        if 'SMA_20' in data.columns:
            prophet_data['sma_20'] = data['SMA_20'].values
        if 'Volatility' in data.columns:
            prophet_data['volatility'] = data['Volatility'].values
            
        return prophet_data.dropna()
    
    def train(self, data):
        """Train the Prophet model"""
        try:
            if not self.prophet_available:
                print("Prophet not available for training")
                return None
            
            # Prepare data for Prophet
            prophet_data = self.prepare_prophet_data(data.copy())
            
            if len(prophet_data) < 30:
                raise ValueError("Insufficient data for Prophet training (minimum 30 days required)")
            
            # Initialize Prophet model with custom parameters for stock prediction
            self.model = Prophet(
                changepoint_prior_scale=0.05,  # Flexibility in trend changes
                seasonality_prior_scale=10.0,  # Seasonality strength
                holidays_prior_scale=10.0,     # Holiday effects
                daily_seasonality=True,        # Daily patterns
                weekly_seasonality=True,       # Weekly patterns
                yearly_seasonality=True,       # Yearly patterns
                interval_width=0.95,           # Prediction intervals
                growth='linear'                # Linear growth trend
            )
            
            # Add additional regressors if available
            if 'volume' in prophet_data.columns:
                self.model.add_regressor('volume')
            if 'rsi' in prophet_data.columns:
                self.model.add_regressor('rsi')
            if 'macd' in prophet_data.columns:
                self.model.add_regressor('macd')
            if 'sma_20' in prophet_data.columns:
                self.model.add_regressor('sma_20')
            if 'volatility' in prophet_data.columns:
                self.model.add_regressor('volatility')
            
            # Fit the model
            self.model.fit(prophet_data)
            
            # Store training data for future predictions
            self.last_training_data = prophet_data.copy()
            
            # Save model
            os.makedirs('models', exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'training_data': self.last_training_data
                }, f)
            
            # Calculate model performance on last 10% of data
            test_size = max(1, len(prophet_data) // 10)
            train_data = prophet_data[:-test_size]
            test_data = prophet_data[-test_size:]
            
            # Train on subset and predict
            temp_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
            )
            
            # Add regressors to temp model
            for col in prophet_data.columns:
                if col not in ['ds', 'y']:
                    temp_model.add_regressor(col)
            
            temp_model.fit(train_data)
            
            # Make predictions for test period
            forecast = temp_model.predict(test_data[['ds'] + [col for col in test_data.columns if col not in ['ds', 'y']]])
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            actual = test_data['y'].values
            predicted = forecast['yhat'].values
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            return {'mape': mape, 'samples': len(prophet_data)}
            
        except Exception as e:
            print(f"Prophet training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved Prophet model"""
        try:
            if not self.prophet_available:
                return False
                
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.last_training_data = saved_data['training_data']
                return True
        except Exception as e:
            print(f"Prophet model loading error: {str(e)}")
        return False
    
    def exponential_smoothing_prediction(self, data):
        """Fallback prediction using exponential smoothing"""
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        prices = data['Close'].values
        
        # Calculate exponentially smoothed value
        smoothed = prices[0]
        for price in prices[1:]:
            smoothed = alpha * price + (1 - alpha) * smoothed
        
        current_price = data['Close'].iloc[-1]
        
        # Trend analysis
        recent_trend = np.mean(np.diff(prices[-10:]))  # Last 10 days trend
        
        # Predict next day price
        predicted_price = smoothed + recent_trend
        
        # Determine direction and confidence
        direction = 'UP' if predicted_price > current_price else 'DOWN'
        price_change_pct = abs((predicted_price - current_price) / current_price) * 100
        confidence = min(75.0, 50.0 + price_change_pct * 10)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'Prophet (Exponential Smoothing Fallback)'
        }
    
    def predict(self, data):
        """Make Prophet prediction for next day"""
        try:
            if not self.prophet_available:
                return self.exponential_smoothing_prediction(data)
            
            # Try to load existing model first
            if self.model is None:
                if not self.load_model():
                    # Train new model if no saved model exists
                    print("Training new Prophet model...")
                    result = self.train(data.copy())
                    if result is None:
                        return self.exponential_smoothing_prediction(data)
            
            # Prepare future dataframe for prediction
            current_data = self.prepare_prophet_data(data.copy())
            
            # Create future dataframe for next day
            future_date = current_data['ds'].iloc[-1] + timedelta(days=1)
            future_df = pd.DataFrame({'ds': [future_date]})
            
            # Add regressor values for future prediction (use last known values)
            for col in current_data.columns:
                if col not in ['ds', 'y']:
                    future_df[col] = current_data[col].iloc[-1]
            
            # Make prediction
            forecast = self.model.predict(future_df)
            
            predicted_price = forecast['yhat'].iloc[0]
            current_price = data['Close'].iloc[-1]
            
            # Calculate confidence intervals
            lower_bound = forecast['yhat_lower'].iloc[0]
            upper_bound = forecast['yhat_upper'].iloc[0]
            
            # Determine direction and confidence
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Calculate confidence based on prediction interval width
            interval_width = (upper_bound - lower_bound) / current_price
            confidence = max(50.0, min(90.0, 100.0 - interval_width * 100))
            
            # Additional trend confirmation
            if len(data) >= 5:
                recent_trend = np.mean(np.diff(data['Close'].iloc[-5:].values))
                trend_direction = 'UP' if recent_trend > 0 else 'DOWN'
                
                # Boost confidence if prediction aligns with recent trend
                if direction == trend_direction:
                    confidence = min(95.0, confidence * 1.1)
                else:
                    confidence = max(40.0, confidence * 0.9)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_type': 'Prophet (Time Series Forecasting)'
            }
            
        except Exception as e:
            print(f"Prophet prediction error: {str(e)}")
            return self.exponential_smoothing_prediction(data)
    
    def predict_multiple_days(self, data, days=7):
        """Predict multiple days ahead"""
        try:
            if not self.prophet_available or self.model is None:
                return None
            
            # Prepare current data
            current_data = self.prepare_prophet_data(data.copy())
            
            # Create future dataframe for multiple days
            future_dates = [current_data['ds'].iloc[-1] + timedelta(days=i) for i in range(1, days + 1)]
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Add regressor values (use last known values)
            for col in current_data.columns:
                if col not in ['ds', 'y']:
                    future_df[col] = current_data[col].iloc[-1]
            
            # Make predictions
            forecast = self.model.predict(future_df)
            
            predictions = []
            for i, (_, row) in enumerate(forecast.iterrows()):
                predictions.append({
                    'date': future_dates[i],
                    'predicted_price': row['yhat'],
                    'lower_bound': row['yhat_lower'],
                    'upper_bound': row['yhat_upper']
                })
            
            return predictions
            
        except Exception as e:
            print(f"Multi-day prediction error: {str(e)}")
            return None