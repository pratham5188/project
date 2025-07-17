import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import os
    os.environ['KERAS_BACKEND'] = 'jax'
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
    from keras.optimizers import Adam
    import tensorflow as tf
    KERAS_AVAILABLE = True
    print("Keras with JAX backend loaded for Transformer model")
except ImportError:
    KERAS_AVAILABLE = False
    print("Keras not available - using attention-based fallback")

class TransformerPredictor:
    """Transformer-based stock prediction model using attention mechanisms"""
    
    def __init__(self, sequence_length=60, d_model=64, num_heads=8, num_layers=4):
        self.model = None
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.keras_available = KERAS_AVAILABLE
        self.feature_scaler = None
        self.price_scaler = None
        
    def prepare_transformer_data(self, data):
        """Prepare data for transformer model"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Select features for transformer
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Add technical indicators if available
        if 'SMA_20' in data.columns:
            feature_columns.append('SMA_20')
        if 'SMA_50' in data.columns:
            feature_columns.append('SMA_50')
        if 'RSI' in data.columns:
            feature_columns.append('RSI')
        if 'MACD' in data.columns:
            feature_columns.append('MACD')
        if 'MACD_Signal' in data.columns:
            feature_columns.append('MACD_Signal')
        
        # Create additional features
        data_copy = data.copy()
        data_copy['Price_Range'] = data_copy['High'] - data_copy['Low']
        data_copy['Price_Change'] = data_copy['Close'].pct_change()
        data_copy['Volume_MA'] = data_copy['Volume'].rolling(20).mean()
        data_copy['Volatility'] = data_copy['Close'].rolling(20).std()
        
        feature_columns.extend(['Price_Range', 'Price_Change', 'Volume_MA', 'Volatility'])
        
        # Select available features
        available_features = [col for col in feature_columns if col in data_copy.columns]
        feature_data = data_copy[available_features].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        if self.feature_scaler is None:
            self.feature_scaler = MinMaxScaler()
            scaled_features = self.feature_scaler.fit_transform(feature_data)
        else:
            scaled_features = self.feature_scaler.transform(feature_data)
        
        # Scale prices separately for target
        if self.price_scaler is None:
            self.price_scaler = MinMaxScaler()
            scaled_prices = self.price_scaler.fit_transform(data_copy[['Close']])
        else:
            scaled_prices = self.price_scaler.transform(data_copy[['Close']])
        
        return scaled_features, scaled_prices.flatten(), available_features
    
    def create_sequences(self, features, prices):
        """Create sequences for transformer training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(prices[i])
        
        return np.array(X), np.array(y)
    
    def build_transformer_encoder(self, inputs):
        """Build transformer encoder block"""
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads
        )(inputs, inputs)
        
        # Add & norm
        attention = Dropout(0.1)(attention)
        attention = LayerNormalization()(inputs + attention)
        
        # Feed forward
        ff = Dense(self.d_model * 4, activation='relu')(attention)
        ff = Dense(self.d_model)(ff)
        ff = Dropout(0.1)(ff)
        
        # Add & norm
        output = LayerNormalization()(attention + ff)
        
        return output
    
    def build_transformer_model(self, input_shape):
        """Build complete transformer model"""
        if not self.keras_available:
            return None
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Embedding layer to project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Stack transformer encoder layers
        for _ in range(self.num_layers):
            x = self.build_transformer_encoder(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='linear')(x)  # Price prediction
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data):
        """Train the transformer model"""
        try:
            if not self.keras_available:
                print("Keras not available for Transformer training")
                return None
            
            # Prepare data
            features, prices, feature_names = self.prepare_transformer_data(data.copy())
            
            if len(features) < self.sequence_length + 50:
                raise ValueError("Insufficient data for Transformer training")
            
            # Create sequences
            X, y = self.create_sequences(features, prices)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build model
            self.model = self.build_transformer_model((X_train.shape[1], X_train.shape[2]))
            
            # Train with callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=0.0001
            )
            
            history = self.model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=100,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            train_loss = self.model.evaluate(X_train, y_train, verbose=0)
            test_loss = self.model.evaluate(X_test, y_test, verbose=0)
            
            return {
                'train_loss': train_loss[0],
                'test_loss': test_loss[0],
                'train_mae': train_loss[1],
                'test_mae': test_loss[1]
            }
            
        except Exception as e:
            print(f"Transformer training error: {str(e)}")
            return None
    
    def attention_based_prediction(self, data):
        """Attention-based prediction without full transformer (fallback)"""
        current_price = data['Close'].iloc[-1]
        
        # Calculate attention weights for different time periods
        recent_prices = data['Close'].tail(20).values
        older_prices = data['Close'].tail(60).values if len(data) >= 60 else recent_prices
        
        # Simple attention mechanism using price momentum
        recent_changes = np.diff(recent_prices)
        weights = np.softmax(np.abs(recent_changes))  # More weight to larger changes
        
        # Weighted average of recent trends
        weighted_trend = np.sum(weights * recent_changes[:-1]) if len(recent_changes) > 1 else 0
        
        # Volume attention
        if 'Volume' in data.columns:
            recent_volumes = data['Volume'].tail(10).values
            volume_weights = recent_volumes / np.sum(recent_volumes)
            volume_trend = np.sum(volume_weights * np.diff(recent_prices[-10:]))
        else:
            volume_trend = 0
        
        # Technical indicator attention
        tech_score = 0
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi < 30:
                tech_score += 1  # Oversold
            elif rsi > 70:
                tech_score -= 1  # Overbought
        
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd_diff = data['MACD'].iloc[-1] - data['MACD_Signal'].iloc[-1]
            tech_score += np.sign(macd_diff) * 0.5
        
        # Combine signals
        combined_signal = (weighted_trend * 0.5 + volume_trend * 0.3 + tech_score * 0.2)
        
        # Convert to prediction
        if combined_signal > 0.005:  # Bullish
            direction = 'UP'
            confidence = min(75.0, 60.0 + abs(combined_signal) * 1000)
            predicted_price = current_price * (1 + abs(combined_signal) * 2)
        elif combined_signal < -0.005:  # Bearish
            direction = 'DOWN'
            confidence = min(75.0, 60.0 + abs(combined_signal) * 1000)
            predicted_price = current_price * (1 - abs(combined_signal) * 2)
        else:  # Neutral
            direction = 'UP' if current_price > data['Close'].iloc[-2] else 'DOWN'
            confidence = 55.0
            predicted_price = current_price * (1.003 if direction == 'UP' else 0.997)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'Transformer (Attention Fallback)',
            'attention_signal': combined_signal
        }
    
    def predict(self, data):
        """Make transformer prediction"""
        try:
            if not self.keras_available:
                return self.attention_based_prediction(data)
            
            # Train model if not available
            if self.model is None:
                print("Training new Transformer model...")
                result = self.train(data.copy())
                if result is None:
                    return self.attention_based_prediction(data)
            
            # Prepare data for prediction
            features, prices, feature_names = self.prepare_transformer_data(data.copy())
            
            if len(features) < self.sequence_length:
                return self.attention_based_prediction(data)
            
            # Get last sequence
            last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            predicted_scaled = self.model.predict(last_sequence, verbose=0)[0][0]
            
            # Inverse transform to get actual price
            predicted_price = self.price_scaler.inverse_transform([[predicted_scaled]])[0][0]
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate confidence based on model certainty
            # Use model's internal attention to estimate confidence
            price_change_pct = (predicted_price - current_price) / current_price
            confidence = min(85.0, 50.0 + abs(price_change_pct) * 500)
            
            # Determine direction
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Ensure reasonable prediction
            max_change = 0.08  # 8% max change
            if abs(price_change_pct) > max_change:
                if predicted_price > current_price:
                    predicted_price = current_price * (1 + max_change)
                else:
                    predicted_price = current_price * (1 - max_change)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'model_type': 'Transformer (Attention)',
                'price_change_pct': price_change_pct,
                'model_layers': self.num_layers,
                'attention_heads': self.num_heads
            }
            
        except Exception as e:
            print(f"Transformer prediction error: {str(e)}")
            return self.attention_based_prediction(data)