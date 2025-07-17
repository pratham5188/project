import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("PyTorch loaded successfully for Transformer models")
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"PyTorch not available - using fallback implementation: {e}")

class TransformerPredictor:
    """Transformer-based stock prediction model using attention mechanisms"""
    
    def __init__(self, sequence_length=60, d_model=64, n_heads=8, n_layers=6, n_features=10):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_features = n_features
        self.model_path = 'models/transformer_model.pth'
        self.scaler_path = 'models/transformer_scaler.pkl'
        self.torch_available = TORCH_AVAILABLE
        self.feature_columns = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        if not self.torch_available:
            print("PyTorch not available, using simplified transformer alternative")
    
    def prepare_features(self, data):
        """Prepare comprehensive features for Transformer training"""
        features_df = pd.DataFrame()
        
        # Basic OHLCV features
        features_df['close'] = data['Close']
        features_df['high'] = data['High']
        features_df['low'] = data['Low']
        features_df['open'] = data['Open']
        features_df['volume'] = data['Volume']
        
        # Price-based features
        features_df['returns'] = data['Close'].pct_change()
        features_df['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features_df['high_low_ratio'] = data['High'] / data['Low']
        features_df['close_open_ratio'] = data['Close'] / data['Open']
        
        # Technical indicators
        if 'RSI' in data.columns:
            features_df['rsi'] = data['RSI'] / 100.0
        else:
            features_df['rsi'] = 0.5
            
        if 'MACD' in data.columns:
            features_df['macd'] = data['MACD']
        else:
            features_df['macd'] = 0.0
            
        if 'MACD_Signal' in data.columns:
            features_df['macd_signal'] = data['MACD_Signal']
        else:
            features_df['macd_signal'] = 0.0
            
        if 'SMA_20' in data.columns:
            features_df['sma_20_ratio'] = data['Close'] / data['SMA_20']
        else:
            features_df['sma_20_ratio'] = 1.0
            
        if 'SMA_50' in data.columns:
            features_df['sma_50_ratio'] = data['Close'] / data['SMA_50']
        else:
            features_df['sma_50_ratio'] = 1.0
            
        if 'Volatility' in data.columns:
            features_df['volatility'] = data['Volatility']
        else:
            features_df['volatility'] = data['Close'].rolling(window=20).std()
        
        # Advanced features
        features_df['price_momentum'] = data['Close'].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        features_df['volume_momentum'] = data['Volume'].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        # Moving averages
        features_df['ma_5'] = data['Close'].rolling(window=5).mean()
        features_df['ma_10'] = data['Close'].rolling(window=10).mean()
        features_df['ma_20'] = data['Close'].rolling(window=20).mean()
        
        # Price position relative to moving averages
        features_df['price_ma5_ratio'] = data['Close'] / features_df['ma_5']
        features_df['price_ma10_ratio'] = data['Close'] / features_df['ma_10']
        features_df['price_ma20_ratio'] = data['Close'] / features_df['ma_20']
        
        # Volume features
        features_df['volume_ma'] = data['Volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = data['Volume'] / features_df['volume_ma']
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            bb_width = data['BB_Upper'] - data['BB_Lower']
            features_df['bb_position'] = (data['Close'] - data['BB_Lower']) / bb_width
            features_df['bb_width'] = bb_width / data['Close']
        else:
            features_df['bb_position'] = 0.5
            features_df['bb_width'] = 0.1
        
        # Select most important features
        self.feature_columns = [
            'close', 'volume', 'returns', 'high_low_ratio', 'rsi', 
            'macd', 'volatility', 'price_momentum', 'volume_ratio', 'bb_position'
        ]
        
        # Ensure we have required features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        if len(available_features) < self.n_features:
            additional = [
                'log_returns', 'close_open_ratio', 'macd_signal', 'sma_20_ratio',
                'volume_momentum', 'price_ma5_ratio', 'bb_width'
            ]
            for col in additional:
                if col in features_df.columns and col not in available_features:
                    available_features.append(col)
                    if len(available_features) >= self.n_features:
                        break
        
        self.feature_columns = available_features[:self.n_features]
        
        return features_df[self.feature_columns].ffill().bfill()

class StockTransformer(nn.Module):
    """Transformer model for stock prediction"""
    
    def __init__(self, n_features, d_model, n_heads, n_layers, sequence_length):
        super(StockTransformer, self).__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(sequence_length, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1)
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_encoding
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(x)
        
        # Use the last timestep for prediction
        last_output = transformer_output[:, -1, :]
        
        # Apply dropout and output projection
        output = self.dropout(last_output)
        prediction = self.output_projection(output)
        
        return prediction

class TransformerPredictor:
    """Transformer-based stock prediction model using attention mechanisms"""
    
    def __init__(self, sequence_length=60, d_model=64, n_heads=8, n_layers=6, n_features=10):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_features = n_features
        self.model_path = 'models/transformer_model.pth'
        self.scaler_path = 'models/transformer_scaler.pkl'
        self.torch_available = TORCH_AVAILABLE
        self.feature_columns = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        
        if not self.torch_available:
            print("PyTorch not available, using simplified transformer alternative")
    
    def prepare_features(self, data):
        """Prepare comprehensive features for Transformer training"""
        features_df = pd.DataFrame()
        
        # Basic OHLCV features
        features_df['close'] = data['Close']
        features_df['high'] = data['High']
        features_df['low'] = data['Low']
        features_df['open'] = data['Open']
        features_df['volume'] = data['Volume']
        
        # Price-based features
        features_df['returns'] = data['Close'].pct_change()
        features_df['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features_df['high_low_ratio'] = data['High'] / data['Low']
        features_df['close_open_ratio'] = data['Close'] / data['Open']
        
        # Technical indicators
        if 'RSI' in data.columns:
            features_df['rsi'] = data['RSI'] / 100.0
        else:
            features_df['rsi'] = 0.5
            
        if 'MACD' in data.columns:
            features_df['macd'] = data['MACD']
        else:
            features_df['macd'] = 0.0
            
        if 'MACD_Signal' in data.columns:
            features_df['macd_signal'] = data['MACD_Signal']
        else:
            features_df['macd_signal'] = 0.0
            
        if 'SMA_20' in data.columns:
            features_df['sma_20_ratio'] = data['Close'] / data['SMA_20']
        else:
            features_df['sma_20_ratio'] = 1.0
            
        if 'SMA_50' in data.columns:
            features_df['sma_50_ratio'] = data['Close'] / data['SMA_50']
        else:
            features_df['sma_50_ratio'] = 1.0
            
        if 'Volatility' in data.columns:
            features_df['volatility'] = data['Volatility']
        else:
            features_df['volatility'] = data['Close'].rolling(window=20).std()
        
        # Advanced features
        features_df['price_momentum'] = data['Close'].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        features_df['volume_momentum'] = data['Volume'].rolling(window=10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )
        
        # Moving averages
        features_df['ma_5'] = data['Close'].rolling(window=5).mean()
        features_df['ma_10'] = data['Close'].rolling(window=10).mean()
        features_df['ma_20'] = data['Close'].rolling(window=20).mean()
        
        # Price position relative to moving averages
        features_df['price_ma5_ratio'] = data['Close'] / features_df['ma_5']
        features_df['price_ma10_ratio'] = data['Close'] / features_df['ma_10']
        features_df['price_ma20_ratio'] = data['Close'] / features_df['ma_20']
        
        # Volume features
        features_df['volume_ma'] = data['Volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = data['Volume'] / features_df['volume_ma']
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            bb_width = data['BB_Upper'] - data['BB_Lower']
            features_df['bb_position'] = (data['Close'] - data['BB_Lower']) / bb_width
            features_df['bb_width'] = bb_width / data['Close']
        else:
            features_df['bb_position'] = 0.5
            features_df['bb_width'] = 0.1
        
        # Select most important features
        self.feature_columns = [
            'close', 'volume', 'returns', 'high_low_ratio', 'rsi', 
            'macd', 'volatility', 'price_momentum', 'volume_ratio', 'bb_position'
        ]
        
        # Ensure we have required features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        if len(available_features) < self.n_features:
            additional = [
                'log_returns', 'close_open_ratio', 'macd_signal', 'sma_20_ratio',
                'volume_momentum', 'price_ma5_ratio', 'bb_width'
            ]
            for col in additional:
                if col in features_df.columns and col not in available_features:
                    available_features.append(col)
                    if len(available_features) >= self.n_features:
                        break
        
        self.feature_columns = available_features[:self.n_features]
        
        return features_df[self.feature_columns].ffill().bfill()
    
    def create_sequences(self, data):
        """Create sequences for Transformer training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict close price (first column)
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the Transformer model"""
        try:
            if not self.torch_available:
                print("PyTorch not available for Transformer training")
                return None
            
            # Prepare features
            features = self.prepare_features(data.copy())
            
            if len(features) < self.sequence_length + 50:
                raise ValueError(f"Insufficient data for Transformer training (minimum {self.sequence_length + 50} days required)")
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(features)
            
            # Create sequences
            X, y = self.create_sequences(scaled_data)
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            self.model = StockTransformer(
                n_features=len(self.feature_columns),
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                sequence_length=self.sequence_length
            ).to(self.device)
            
            # Optimizer and loss function
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                train_losses.append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    os.makedirs('models', exist_ok=True)
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'feature_columns': self.feature_columns,
                        'sequence_length': self.sequence_length,
                        'd_model': self.d_model,
                        'n_heads': self.n_heads,
                        'n_layers': self.n_layers
                    }, self.model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= 20:  # Early stopping
                        break
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'sequence_length': self.sequence_length
                }, f)
            
            # Calculate final metrics
            self.model.eval()
            with torch.no_grad():
                train_pred = self.model(X_train_tensor).squeeze().cpu().numpy()
                val_pred = self.model(X_val_tensor).squeeze().cpu().numpy()
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            
            return {
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'epochs_trained': epoch + 1,
                'best_val_loss': best_val_loss
            }
            
        except Exception as e:
            print(f"Transformer training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved Transformer model"""
        try:
            if not self.torch_available:
                return False
                
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Load model checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Initialize model with saved parameters
                self.model = StockTransformer(
                    n_features=len(checkpoint['feature_columns']),
                    d_model=checkpoint['d_model'],
                    n_heads=checkpoint['n_heads'],
                    n_layers=checkpoint['n_layers'],
                    sequence_length=checkpoint['sequence_length']
                ).to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Load scaler and features
                with open(self.scaler_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.scaler = saved_data['scaler']
                    self.feature_columns = saved_data['feature_columns']
                    self.sequence_length = saved_data['sequence_length']
                
                return True
        except Exception as e:
            print(f"Transformer model loading error: {str(e)}")
        return False
    
    def attention_based_prediction(self, data):
        """Fallback prediction using attention-like weighted averages"""
        prices = data['Close'].values
        
        # Create attention weights (more weight to recent prices)
        sequence_len = min(30, len(prices))
        recent_prices = prices[-sequence_len:]
        
        # Exponential attention weights
        attention_weights = np.exp(np.arange(sequence_len) / 5)
        attention_weights = attention_weights / attention_weights.sum()
        
        # Weighted prediction
        weighted_price = np.sum(recent_prices * attention_weights)
        
        current_price = data['Close'].iloc[-1]
        
        # Trend analysis with attention
        if len(prices) >= 10:
            short_trend = np.mean(np.diff(prices[-5:]))
            long_trend = np.mean(np.diff(prices[-10:]))
            trend_weight = 0.7 * short_trend + 0.3 * long_trend
        else:
            trend_weight = np.mean(np.diff(prices[-len(prices)+1:]))
        
        # Predict next price
        predicted_price = weighted_price + trend_weight
        
        # Direction and confidence
        direction = 'UP' if predicted_price > current_price else 'DOWN'
        price_change_pct = abs((predicted_price - current_price) / current_price) * 100
        
        # Confidence based on trend consistency
        recent_changes = np.diff(prices[-5:]) if len(prices) >= 5 else np.diff(prices)
        trend_consistency = len([x for x in recent_changes if (x > 0) == (trend_weight > 0)]) / len(recent_changes)
        confidence = 50.0 + trend_consistency * 25.0 + min(price_change_pct * 5, 20.0)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'Transformer (Attention Fallback)'
        }
    
    def predict(self, data):
        """Make Transformer prediction for next day"""
        try:
            if not self.torch_available:
                return self.attention_based_prediction(data)
            
            # Try to load existing model first
            if self.model is None:
                if not self.load_model():
                    # Train new model if no saved model exists
                    print("Training new Transformer model...")
                    result = self.train(data.copy())
                    if result is None:
                        return self.attention_based_prediction(data)
            
            # Prepare features
            features = self.prepare_features(data.copy())
            
            if len(features) < self.sequence_length:
                return self.attention_based_prediction(data)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Get last sequence
            last_sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                predicted_scaled = self.model(last_sequence_tensor).cpu().numpy()[0][0]
            
            # Inverse transform prediction
            dummy_array = np.zeros((1, len(self.feature_columns)))
            dummy_array[0, 0] = predicted_scaled
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            current_price = data['Close'].iloc[-1]
            
            # Calculate direction and confidence
            price_change = (predicted_price - current_price) / current_price
            direction = 'UP' if predicted_price > current_price else 'DOWN'
            
            # Enhanced confidence calculation
            confidence = 65.0  # Base confidence for Transformer
            
            # Adjust confidence based on prediction magnitude
            price_change_magnitude = abs(price_change)
            if price_change_magnitude > 0.05:  # > 5% change
                confidence = min(90.0, confidence + 20.0)
            elif price_change_magnitude > 0.02:  # > 2% change
                confidence = min(85.0, confidence + 15.0)
            elif price_change_magnitude > 0.01:  # > 1% change
                confidence = min(80.0, confidence + 10.0)
            
            # Recent trend confirmation
            if len(data) >= 5:
                recent_trend = np.mean(np.diff(data['Close'].iloc[-5:].values))
                trend_direction = 'UP' if recent_trend > 0 else 'DOWN'
                
                if direction == trend_direction:
                    confidence = min(95.0, confidence * 1.15)
                else:
                    confidence = max(50.0, confidence * 0.85)
            
            # Volume and volatility adjustments
            if 'Volume' in data.columns and 'Volatility' in data.columns:
                recent_volume = np.mean(data['Volume'].iloc[-3:])
                avg_volume = np.mean(data['Volume'].iloc[-10:])
                recent_volatility = data['Volatility'].iloc[-1] if not pd.isna(data['Volatility'].iloc[-1]) else 0.02
                
                volume_support = recent_volume / avg_volume if avg_volume > 0 else 1.0
                
                # High volume and low volatility increase confidence
                if volume_support > 1.3 and recent_volatility < 0.02:
                    confidence = min(98.0, confidence * 1.1)
                elif volume_support < 0.7 or recent_volatility > 0.05:
                    confidence = max(40.0, confidence * 0.9)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'price_change_pct': price_change * 100,
                'model_type': 'Transformer (Attention Mechanism)'
            }
            
        except Exception as e:
            print(f"Transformer prediction error: {str(e)}")
            return self.attention_based_prediction(data)