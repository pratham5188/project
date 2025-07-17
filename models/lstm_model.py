import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow is not available in this environment - using fallback implementation
TENSORFLOW_AVAILABLE = False

class LSTMPredictor:
    """LSTM-based stock prediction model for deep learning analysis"""
    
    def __init__(self, sequence_length=60):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.model_path = 'models/lstm_model.h5'
        self.scaler_path = 'models/lstm_scaler.pkl'
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        
        if not self.tensorflow_available:
            print("TensorFlow not available, using simplified LSTM alternative")
    
    def prepare_lstm_data(self, data):
        """Prepare data for LSTM training"""
        # Use multiple features for LSTM
        feature_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
        if 'SMA_20' in data.columns:
            feature_columns.append('SMA_20')
        if 'RSI' in data.columns:
            feature_columns.append('RSI')
        if 'MACD' in data.columns:
            feature_columns.append('MACD')
        
        # Select available features
        available_features = [col for col in feature_columns if col in data.columns]
        feature_data = data[available_features].ffill().bfill()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        return scaled_data, available_features
    
    def create_sequences(self, data, target_column_index=0):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_column_index])  # Predict close price
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        if not self.tensorflow_available:
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, data):
        """Train the LSTM model"""
        try:
            if not self.tensorflow_available:
                print("TensorFlow not available for LSTM training")
                return None
            
            # Prepare data
            scaled_data, feature_columns = self.prepare_lstm_data(data.copy())
            
            if len(scaled_data) < self.sequence_length + 50:
                raise ValueError("Insufficient data for LSTM training")
            
            # Create sequences
            X, y = self.create_sequences(scaled_data)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=50,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate model
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Save model and scaler
            os.makedirs('models', exist_ok=True)
            self.model.save(self.model_path)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            return {'train_rmse': train_rmse, 'test_rmse': test_rmse}
            
        except Exception as e:
            print(f"LSTM training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved LSTM model and scaler"""
        try:
            if not self.tensorflow_available:
                return False
                
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = keras.models.load_model(self.model_path)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
        except Exception as e:
            print(f"LSTM model loading error: {str(e)}")
        return False
    
    def simple_moving_average_prediction(self, data):
        """Fallback prediction using moving averages"""
        # Calculate trend using multiple moving averages
        short_ma = data['Close'].rolling(window=5).mean().iloc[-1]
        long_ma = data['Close'].rolling(window=20).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        # Simple trend analysis
        if short_ma > long_ma and current_price > short_ma:
            direction = 'UP'
            confidence = 65.0
            predicted_price = current_price * 1.02  # 2% increase
        elif short_ma < long_ma and current_price < short_ma:
            direction = 'DOWN'
            confidence = 65.0
            predicted_price = current_price * 0.98  # 2% decrease
        else:
            direction = 'UP' if current_price > data['Close'].iloc[-2] else 'DOWN'
            confidence = 55.0
            predicted_price = current_price * (1.01 if direction == 'UP' else 0.99)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'LSTM (Moving Average Fallback)'
        }
    
    def predict(self, data):
        """Make LSTM prediction for next day"""
        try:
            if not self.tensorflow_available:
                return self.simple_moving_average_prediction(data)
            
            # Try to load existing model first
            if self.model is None:
                if not self.load_model():
                    # Train new model if no saved model exists
                    print("Training new LSTM model...")
                    result = self.train(data.copy())
                    if result is None:
                        return self.simple_moving_average_prediction(data)
            
            # Prepare data for prediction
            scaled_data, feature_columns = self.prepare_lstm_data(data.copy())
            
            if len(scaled_data) < self.sequence_length:
                return self.simple_moving_average_prediction(data)
            
            # Get last sequence
            last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            predicted_scaled = self.model.predict(last_sequence, verbose=0)[0][0]
            
            # Inverse transform prediction (for close price)
            current_price = data['Close'].iloc[-1]
            
            # Create dummy array for inverse transform
            dummy_array = np.zeros((1, len(feature_columns)))
            dummy_array[0, 0] = predicted_scaled  # Close price is first feature
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            # Determine direction and confidence
            price_change = (predicted_price - current_price) / current_price
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Calculate confidence based on prediction magnitude
            confidence = min(80.0, 50.0 + abs(price_change) * 1000)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'model_type': 'LSTM (Deep Learning)'
            }
            
        except Exception as e:
            print(f"LSTM prediction error: {str(e)}")
            return self.simple_moving_average_prediction(data)
