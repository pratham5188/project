import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
    import keras
    from keras.models import Sequential
    from keras.layers import GRU, Dense, Dropout, BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    KERAS_AVAILABLE = True
    print("Using Keras 3 with JAX backend for GRU models")
except ImportError as e:
    KERAS_AVAILABLE = False
    print(f"Keras/JAX not available - using fallback implementation: {e}")

class GRUPredictor:
    """GRU-based stock prediction model for advanced sequence learning"""
    
    def __init__(self, sequence_length=60, n_features=8):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_path = 'models/gru_model.h5'
        self.scaler_path = 'models/gru_scaler.pkl'
        self.keras_available = KERAS_AVAILABLE
        self.feature_columns = []
        
        if not self.keras_available:
            print("Keras not available, using simplified GRU alternative")
    
    def prepare_features(self, data):
        """Prepare enhanced features for GRU training"""
        # Base price features
        features_df = pd.DataFrame()
        features_df['close'] = data['Close']
        features_df['high'] = data['High']
        features_df['low'] = data['Low']
        features_df['open'] = data['Open']
        features_df['volume'] = data['Volume']
        
        # Price-based features
        features_df['price_change'] = data['Close'].pct_change()
        features_df['high_low_ratio'] = data['High'] / data['Low']
        features_df['close_open_ratio'] = data['Close'] / data['Open']
        
        # Technical indicators
        if 'RSI' in data.columns:
            features_df['rsi'] = data['RSI'] / 100.0  # Normalize RSI
        else:
            features_df['rsi'] = 0.5  # Neutral RSI
            
        if 'MACD' in data.columns:
            features_df['macd'] = data['MACD']
        else:
            features_df['macd'] = 0.0
            
        if 'SMA_20' in data.columns:
            features_df['sma_ratio'] = data['Close'] / data['SMA_20']
        else:
            features_df['sma_ratio'] = 1.0
            
        if 'Volatility' in data.columns:
            features_df['volatility'] = data['Volatility']
        else:
            # Calculate volatility manually
            features_df['volatility'] = data['Close'].rolling(window=20).std()
        
        # Moving averages
        features_df['ma_5'] = data['Close'].rolling(window=5).mean()
        features_df['ma_10'] = data['Close'].rolling(window=10).mean()
        features_df['ma_20'] = data['Close'].rolling(window=20).mean()
        
        # Price position indicators
        features_df['price_position'] = (data['Close'] - features_df['ma_20']) / features_df['ma_20']
        
        # Volume indicators
        features_df['volume_ma'] = data['Volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = data['Volume'] / features_df['volume_ma']
        
        # Select top features for GRU
        self.feature_columns = [
            'close', 'volume', 'price_change', 'high_low_ratio', 
            'rsi', 'macd', 'volatility', 'price_position'
        ]
        
        # Ensure we have the required number of features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        if len(available_features) < self.n_features:
            # Add additional computed features if needed
            additional = ['close_open_ratio', 'sma_ratio', 'volume_ratio']
            for col in additional:
                if col in features_df.columns and col not in available_features:
                    available_features.append(col)
                    if len(available_features) >= self.n_features:
                        break
        
        self.feature_columns = available_features[:self.n_features]
        
        return features_df[self.feature_columns].ffill().bfill()
    
    def create_sequences(self, data):
        """Create sequences for GRU training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict close price (first column)
        return np.array(X), np.array(y)
    
    def build_gru_model(self, input_shape):
        """Build advanced GRU model architecture"""
        if not self.keras_available:
            return None
        
        model = Sequential([
            # First GRU layer with return sequences
            GRU(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second GRU layer with return sequences
            GRU(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third GRU layer without return sequences
            GRU(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            # Dense layers
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, data):
        """Train the GRU model"""
        try:
            if not self.keras_available:
                print("Keras not available for GRU training")
                return None
            
            # Prepare features
            features = self.prepare_features(data.copy())
            
            if len(features) < self.sequence_length + 50:
                raise ValueError(f"Insufficient data for GRU training (minimum {self.sequence_length + 50} days required)")
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(features)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data)
            
            # Split data (80% train, 20% validation)
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Build model
            self.model = self.build_gru_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001,
                verbose=0
            )
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=100,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            train_pred = self.model.predict(X_train, verbose=0)
            val_pred = self.model.predict(X_val, verbose=0)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            # Save model and scaler
            os.makedirs('models', exist_ok=True)
            self.model.save(self.model_path)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'sequence_length': self.sequence_length
                }, f)
            
            return {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'epochs_trained': len(history.history['loss'])
            }
            
        except Exception as e:
            print(f"GRU training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved GRU model and scaler"""
        try:
            if not self.keras_available:
                return False
                
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = keras.models.load_model(self.model_path)
                with open(self.scaler_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.scaler = saved_data['scaler']
                    self.feature_columns = saved_data['feature_columns']
                    self.sequence_length = saved_data['sequence_length']
                return True
        except Exception as e:
            print(f"GRU model loading error: {str(e)}")
        return False
    
    def weighted_moving_average_prediction(self, data):
        """Fallback prediction using weighted moving average"""
        # Calculate weighted moving average with more weight on recent prices
        prices = data['Close'].values
        weights = np.arange(1, len(prices) + 1)
        wma = np.average(prices, weights=weights)
        
        current_price = data['Close'].iloc[-1]
        
        # Momentum analysis
        momentum = np.mean(np.diff(prices[-10:]))  # Last 10 days momentum
        
        # Volume trend
        volume_trend = 1.0
        if 'Volume' in data.columns:
            recent_volume = np.mean(data['Volume'].iloc[-5:])
            avg_volume = np.mean(data['Volume'].iloc[-20:])
            volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Predict next day price
        predicted_price = wma + momentum * volume_trend
        
        # Determine direction and confidence
        direction = 'UP' if predicted_price > current_price else 'DOWN'
        price_change_pct = abs((predicted_price - current_price) / current_price) * 100
        
        # Confidence based on momentum consistency
        recent_changes = np.diff(prices[-5:])
        momentum_consistency = len([x for x in recent_changes if (x > 0) == (momentum > 0)]) / len(recent_changes)
        confidence = 50.0 + momentum_consistency * 30.0 + min(price_change_pct * 10, 20.0)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'GRU (Weighted MA Fallback)'
        }
    
    def predict(self, data):
        """Make GRU prediction for next day"""
        try:
            if not self.keras_available:
                return self.weighted_moving_average_prediction(data)
            
            # Try to load existing model first
            if self.model is None:
                if not self.load_model():
                    # Train new model if no saved model exists
                    print("Training new GRU model...")
                    result = self.train(data.copy())
                    if result is None:
                        return self.weighted_moving_average_prediction(data)
            
            # Prepare features
            features = self.prepare_features(data.copy())
            
            if len(features) < self.sequence_length:
                return self.weighted_moving_average_prediction(data)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get last sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            predicted_scaled = self.model.predict(last_sequence, verbose=0)[0][0]
            
            # Inverse transform prediction
            # Create dummy array for inverse transform
            dummy_array = np.zeros((1, len(self.feature_columns)))
            dummy_array[0, 0] = predicted_scaled  # Close price is first feature
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate direction and confidence
            price_change = (predicted_price - current_price) / current_price
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Enhanced confidence calculation
            confidence = 60.0  # Base confidence for GRU
            
            # Adjust confidence based on prediction magnitude
            price_change_magnitude = abs(price_change)
            if price_change_magnitude > 0.05:  # > 5% change
                confidence = min(85.0, confidence + 15.0)
            elif price_change_magnitude > 0.02:  # > 2% change
                confidence = min(80.0, confidence + 10.0)
            
            # Recent trend confirmation
            if len(data) >= 5:
                recent_trend = np.mean(np.diff(data['Close'].iloc[-5:].values))
                trend_direction = 'UP' if recent_trend > 0 else 'DOWN'
                
                if direction == trend_direction:
                    confidence = min(90.0, confidence * 1.1)
                else:
                    confidence = max(50.0, confidence * 0.9)
            
            # Volume confirmation
            if 'Volume' in data.columns and len(data) >= 10:
                recent_volume = np.mean(data['Volume'].iloc[-3:])
                avg_volume = np.mean(data['Volume'].iloc[-10:])
                volume_support = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                if volume_support > 1.2:  # High volume support
                    confidence = min(95.0, confidence * 1.05)
                elif volume_support < 0.8:  # Low volume
                    confidence = max(45.0, confidence * 0.95)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'price_change_pct': price_change * 100,
                'model_type': 'GRU (Advanced Sequence Learning)'
            }
            
        except Exception as e:
            print(f"GRU prediction error: {str(e)}")
            return self.weighted_moving_average_prediction(data)
    
    def predict_sequence(self, data, steps=5):
        """Predict multiple steps ahead using iterative approach"""
        try:
            if not self.keras_available or self.model is None:
                return None
            
            # Prepare features
            features = self.prepare_features(data.copy())
            scaled_features = self.scaler.transform(features)
            
            # Start with last sequence
            current_sequence = scaled_features[-self.sequence_length:].copy()
            predictions = []
            
            for step in range(steps):
                # Reshape for prediction
                sequence_input = current_sequence.reshape(1, self.sequence_length, -1)
                
                # Predict next value
                next_pred = self.model.predict(sequence_input, verbose=0)[0][0]
                
                # Store prediction
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                # Remove first value and add prediction
                new_row = current_sequence[-1].copy()
                new_row[0] = next_pred  # Update close price
                
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Inverse transform predictions
            inverse_predictions = []
            for pred in predictions:
                dummy_array = np.zeros((1, len(self.feature_columns)))
                dummy_array[0, 0] = pred
                inverse_pred = self.scaler.inverse_transform(dummy_array)[0, 0]
                inverse_predictions.append(inverse_pred)
            
            return inverse_predictions
            
        except Exception as e:
            print(f"GRU sequence prediction error: {str(e)}")
            return None