import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
from models.xgboost_model import XGBoostPredictor
from models.lstm_model import LSTMPredictor
from models.prophet_model import ProphetPredictor
from models.gru_model import GRUPredictor
from models.transformer_model import TransformerPredictor

class EnsemblePredictor:
    """Advanced ensemble model using stacking techniques for superior predictions"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.model_path = 'models/ensemble_model.pkl'
        self.weights = {}
        self.performance_metrics = {}
        
        # Initialize base models
        self._initialize_base_models()
        
        # Meta-learner models (for stacking)
        self.meta_learners = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=50, random_state=42)
        }
        
        self.best_meta_learner = None
    
    def _initialize_base_models(self):
        """Initialize all base prediction models"""
        try:
            self.base_models = {
                'xgboost': XGBoostPredictor(),
                'lstm': LSTMPredictor(),
                'prophet': ProphetPredictor(),
                'gru': GRUPredictor(),
                'transformer': TransformerPredictor()
            }
            print("All base models initialized successfully")
        except Exception as e:
            print(f"Error initializing base models: {str(e)}")
            # Fallback to simpler models
            self.base_models = {
                'xgboost': XGBoostPredictor(),
                'lstm': LSTMPredictor()
            }
    
    def prepare_ensemble_features(self, data):
        """Prepare comprehensive features for ensemble learning"""
        features_df = pd.DataFrame()
        
        # Basic price features
        features_df['close'] = data['Close']
        features_df['high'] = data['High']
        features_df['low'] = data['Low']
        features_df['open'] = data['Open']
        features_df['volume'] = data['Volume']
        
        # Price-based features
        features_df['returns'] = data['Close'].pct_change()
        features_df['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features_df['high_low_spread'] = (data['High'] - data['Low']) / data['Close']
        features_df['open_close_spread'] = (data['Close'] - data['Open']) / data['Open']
        
        # Technical indicators (if available)
        tech_indicators = ['RSI', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower', 'Volatility']
        for indicator in tech_indicators:
            if indicator in data.columns:
                features_df[indicator.lower()] = data[indicator]
        
        # Advanced features
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(data) >= window:
                features_df[f'ma_{window}'] = data['Close'].rolling(window=window).mean()
                features_df[f'ma_{window}_ratio'] = data['Close'] / features_df[f'ma_{window}']
        
        # Momentum indicators
        for window in [5, 10, 20]:
            if len(data) >= window:
                features_df[f'momentum_{window}'] = data['Close'].rolling(window=window).apply(
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
                )
        
        # Volatility measures
        for window in [10, 20, 30]:
            if len(data) >= window:
                features_df[f'volatility_{window}'] = data['Close'].rolling(window=window).std()
                features_df[f'volatility_{window}_norm'] = features_df[f'volatility_{window}'] / data['Close']
        
        # Volume indicators
        if len(data) >= 20:
            features_df['volume_ma_20'] = data['Volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = data['Volume'] / features_df['volume_ma_20']
            features_df['price_volume'] = data['Close'] * data['Volume']
        
        # Trend indicators
        if len(data) >= 10:
            features_df['trend_5'] = data['Close'].rolling(window=5).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0
            )
            features_df['trend_10'] = data['Close'].rolling(window=10).apply(
                lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0
            )
        
        # Bollinger Bands derived features
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            bb_width = data['BB_Upper'] - data['BB_Lower']
            features_df['bb_position'] = (data['Close'] - data['BB_Lower']) / bb_width
            features_df['bb_width_norm'] = bb_width / data['Close']
            features_df['bb_squeeze'] = bb_width / data['Close'].rolling(window=20).std()
        
        return features_df.ffill().bfill().fillna(0)
    
    def get_base_predictions(self, data):
        """Get predictions from all base models"""
        base_predictions = {}
        prediction_features = []
        
        for model_name, model in self.base_models.items():
            try:
                prediction = model.predict(data.copy())
                if prediction and isinstance(prediction, dict):
                    base_predictions[model_name] = prediction
                    
                    # Extract numerical features for meta-learning
                    prediction_features.extend([
                        prediction.get('predicted_price', 0),
                        prediction.get('confidence', 50) / 100.0,
                        1 if prediction.get('direction', 'UP') == 'UP' else 0
                    ])
                else:
                    # Fallback prediction
                    current_price = data['Close'].iloc[-1]
                    base_predictions[model_name] = {
                        'predicted_price': current_price,
                        'confidence': 50.0,
                        'direction': 'UP',
                        'model_type': f'{model_name} (fallback)'
                    }
                    prediction_features.extend([current_price, 0.5, 1])
                    
            except Exception as e:
                print(f"Error getting prediction from {model_name}: {str(e)}")
                # Fallback prediction
                current_price = data['Close'].iloc[-1]
                base_predictions[model_name] = {
                    'predicted_price': current_price,
                    'confidence': 30.0,
                    'direction': 'UP',
                    'model_type': f'{model_name} (error fallback)'
                }
                prediction_features.extend([current_price, 0.3, 1])
        
        return base_predictions, np.array(prediction_features)
    
    def calculate_dynamic_weights(self, base_predictions, data):
        """Calculate dynamic weights based on recent model performance and market conditions"""
        weights = {}
        current_price = data['Close'].iloc[-1]
        
        # Market condition analysis
        volatility = data['Close'].rolling(window=20).std().iloc[-1] / current_price
        trend_strength = abs(np.mean(np.diff(data['Close'].iloc[-10:].values))) / current_price
        volume_trend = 1.0
        
        if 'Volume' in data.columns and len(data) >= 20:
            recent_volume = np.mean(data['Volume'].iloc[-5:])
            avg_volume = np.mean(data['Volume'].iloc[-20:])
            volume_trend = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        for model_name, prediction in base_predictions.items():
            confidence = prediction.get('confidence', 50.0) / 100.0
            
            # Base weight from confidence
            weight = confidence
            
            # Adjust based on market conditions
            if model_name == 'prophet':
                # Prophet works better in trending markets
                weight *= (1.0 + trend_strength * 0.5)
            elif model_name in ['lstm', 'gru', 'transformer']:
                # Deep learning models work better in complex patterns
                weight *= (1.0 + volatility * 0.3)
            elif model_name == 'xgboost':
                # Tree-based models are generally robust
                weight *= 1.1
            
            # Volume confirmation
            if volume_trend > 1.2:  # High volume
                weight *= 1.05
            elif volume_trend < 0.8:  # Low volume
                weight *= 0.95
            
            weights[model_name] = max(0.1, min(2.0, weight))  # Clamp weights
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights fallback
            weights = {k: 1.0 / len(base_predictions) for k in base_predictions.keys()}
        
        return weights
    
    def train_meta_learner(self, data, lookback_days=100):
        """Train meta-learner using historical predictions"""
        try:
            if len(data) < lookback_days:
                print(f"Insufficient data for meta-learner training. Need {lookback_days}, have {len(data)}")
                return None
            
            # Prepare training data for meta-learner
            meta_features = []
            meta_targets = []
            
            # Use sliding window approach
            for i in range(lookback_days, len(data) - 1):
                historical_data = data.iloc[:i+1].copy()
                
                # Get base model predictions
                _, pred_features = self.get_base_predictions(historical_data)
                
                # Add market features
                market_features = self.prepare_ensemble_features(historical_data).iloc[-1].values
                
                # Combine features
                combined_features = np.concatenate([pred_features, market_features])
                meta_features.append(combined_features)
                
                # Target is next day's actual price
                meta_targets.append(data['Close'].iloc[i+1])
            
            if len(meta_features) < 10:
                print("Insufficient samples for meta-learner training")
                return None
            
            meta_features = np.array(meta_features)
            meta_targets = np.array(meta_targets)
            
            # Handle NaN values
            finite_mask = np.isfinite(meta_features).all(axis=1) & np.isfinite(meta_targets)
            meta_features = meta_features[finite_mask]
            meta_targets = meta_targets[finite_mask]
            
            if len(meta_features) < 5:
                print("Too few valid samples after cleaning")
                return None
            
            # Scale features
            meta_features_scaled = self.scaler.fit_transform(meta_features)
            
            # Train and evaluate different meta-learners
            best_score = -np.inf
            best_meta_learner = None
            
            # Use cross-validation to select best meta-learner
            kfold = KFold(n_splits=min(5, len(meta_features)), shuffle=True, random_state=42)
            
            for name, model in self.meta_learners.items():
                try:
                    scores = cross_val_score(model, meta_features_scaled, meta_targets, 
                                           cv=kfold, scoring='neg_mean_squared_error')
                    avg_score = scores.mean()
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_meta_learner = name
                        
                except Exception as e:
                    print(f"Error training meta-learner {name}: {str(e)}")
                    continue
            
            if best_meta_learner:
                # Train the best meta-learner on all data
                self.meta_model = self.meta_learners[best_meta_learner]
                self.meta_model.fit(meta_features_scaled, meta_targets)
                self.best_meta_learner = best_meta_learner
                
                # Calculate performance metrics
                predictions = self.meta_model.predict(meta_features_scaled)
                rmse = np.sqrt(mean_squared_error(meta_targets, predictions))
                mae = mean_absolute_error(meta_targets, predictions)
                r2 = r2_score(meta_targets, predictions)
                
                self.performance_metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'best_meta_learner': best_meta_learner,
                    'samples_used': len(meta_features)
                }
                
                return self.performance_metrics
            else:
                print("No suitable meta-learner found")
                return None
                
        except Exception as e:
            print(f"Meta-learner training error: {str(e)}")
            return None
    
    def train(self, data):
        """Train the ensemble model"""
        try:
            # Train individual base models
            base_results = {}
            for model_name, model in self.base_models.items():
                try:
                    print(f"Training {model_name} model...")
                    result = model.train(data.copy())
                    base_results[model_name] = result
                    print(f"{model_name} training completed")
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    base_results[model_name] = None
            
            # Train meta-learner
            print("Training meta-learner...")
            meta_result = self.train_meta_learner(data.copy())
            
            # Save ensemble model
            self.save_model()
            
            return {
                'base_models': base_results,
                'meta_learner': meta_result,
                'total_models': len(self.base_models)
            }
            
        except Exception as e:
            print(f"Ensemble training error: {str(e)}")
            return None
    
    def save_model(self):
        """Save the ensemble model"""
        try:
            os.makedirs('models', exist_ok=True)
            ensemble_data = {
                'meta_model': self.meta_model,
                'scaler': self.scaler,
                'weights': self.weights,
                'performance_metrics': self.performance_metrics,
                'best_meta_learner': self.best_meta_learner
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(ensemble_data, f)
            
            print("Ensemble model saved successfully")
            return True
            
        except Exception as e:
            print(f"Error saving ensemble model: {str(e)}")
            return False
    
    def load_model(self):
        """Load the ensemble model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    ensemble_data = pickle.load(f)
                
                self.meta_model = ensemble_data.get('meta_model')
                self.scaler = ensemble_data.get('scaler', StandardScaler())
                self.weights = ensemble_data.get('weights', {})
                self.performance_metrics = ensemble_data.get('performance_metrics', {})
                self.best_meta_learner = ensemble_data.get('best_meta_learner')
                
                return True
        except Exception as e:
            print(f"Error loading ensemble model: {str(e)}")
        return False
    
    def weighted_average_prediction(self, base_predictions, weights):
        """Calculate weighted average prediction from base models"""
        total_predicted_price = 0
        total_confidence = 0
        up_votes = 0
        total_weight = 0
        
        for model_name, prediction in base_predictions.items():
            weight = weights.get(model_name, 1.0 / len(base_predictions))
            
            total_predicted_price += prediction.get('predicted_price', 0) * weight
            total_confidence += prediction.get('confidence', 50) * weight
            
            if prediction.get('direction', 'UP') == 'UP':
                up_votes += weight
            
            total_weight += weight
        
        if total_weight > 0:
            avg_predicted_price = total_predicted_price / total_weight
            avg_confidence = total_confidence / total_weight
            direction = 'UP' if up_votes > (total_weight / 2) else 'DOWN'
        else:
            # Fallback
            avg_predicted_price = list(base_predictions.values())[0].get('predicted_price', 0)
            avg_confidence = 50.0
            direction = 'UP'
        
        return avg_predicted_price, avg_confidence, direction
    
    def predict(self, data):
        """Make ensemble prediction"""
        try:
            # Load model if not already loaded
            if self.meta_model is None:
                self.load_model()
            
            # Get base model predictions
            base_predictions, pred_features = self.get_base_predictions(data)
            
            if not base_predictions:
                # Fallback to simple prediction
                current_price = data['Close'].iloc[-1]
                return {
                    'direction': 'UP',
                    'confidence': 50.0,
                    'predicted_price': current_price,
                    'model_type': 'Ensemble (Fallback)'
                }
            
            # Calculate dynamic weights
            weights = self.calculate_dynamic_weights(base_predictions, data)
            
            # Method 1: Weighted average of base predictions
            weighted_price, weighted_confidence, weighted_direction = self.weighted_average_prediction(
                base_predictions, weights
            )
            
            # Method 2: Meta-learner prediction (if available)
            meta_predicted_price = None
            if self.meta_model is not None:
                try:
                    # Prepare features for meta-learner
                    market_features = self.prepare_ensemble_features(data).iloc[-1].values
                    combined_features = np.concatenate([pred_features, market_features])
                    
                    # Handle NaN values
                    if np.any(np.isnan(combined_features)) or np.any(np.isinf(combined_features)):
                        combined_features = np.nan_to_num(combined_features, 0)
                    
                    combined_features_scaled = self.scaler.transform(combined_features.reshape(1, -1))
                    meta_predicted_price = self.meta_model.predict(combined_features_scaled)[0]
                    
                except Exception as e:
                    print(f"Meta-learner prediction error: {str(e)}")
                    meta_predicted_price = None
            
            # Combine weighted average and meta-learner predictions
            if meta_predicted_price is not None:
                # Give more weight to meta-learner if it has good performance
                meta_weight = 0.6 if self.performance_metrics.get('r2', 0) > 0.5 else 0.3
                weighted_weight = 1.0 - meta_weight
                
                final_predicted_price = (
                    weighted_price * weighted_weight + 
                    meta_predicted_price * meta_weight
                )
                
                # Boost confidence if both methods agree on direction
                current_price = data['Close'].iloc[-1]
                meta_direction = 'UP' if meta_predicted_price > current_price else 'DOWN'
                
                if meta_direction == weighted_direction:
                    final_confidence = min(95.0, weighted_confidence * 1.15)
                else:
                    final_confidence = max(40.0, weighted_confidence * 0.85)
                
                final_direction = weighted_direction  # Use weighted average direction
                
            else:
                # Use only weighted average
                final_predicted_price = weighted_price
                final_confidence = weighted_confidence
                final_direction = weighted_direction
            
            # Additional confidence adjustments
            current_price = data['Close'].iloc[-1]
            price_change_pct = abs((final_predicted_price - current_price) / current_price) * 100
            
            # Boost confidence for ensemble
            final_confidence = min(98.0, final_confidence * 1.1)
            
            # Model agreement boost
            directions = [pred.get('direction', 'UP') for pred in base_predictions.values()]
            direction_consensus = directions.count(final_direction) / len(directions)
            if direction_consensus > 0.7:
                final_confidence = min(98.0, final_confidence * 1.05)
            
            return {
                'direction': final_direction,
                'confidence': final_confidence,
                'predicted_price': final_predicted_price,
                'price_change_pct': (final_predicted_price - current_price) / current_price * 100,
                'model_type': 'Ensemble (Stacking)',
                'base_predictions': base_predictions,
                'weights': weights,
                'meta_used': meta_predicted_price is not None,
                'consensus': direction_consensus
            }
            
        except Exception as e:
            print(f"Ensemble prediction error: {str(e)}")
            # Ultimate fallback
            current_price = data['Close'].iloc[-1]
            recent_trend = np.mean(np.diff(data['Close'].iloc[-5:].values))
            direction = 'UP' if recent_trend > 0 else 'DOWN'
            predicted_price = current_price * (1.01 if direction == 'UP' else 0.99)
            
            return {
                'direction': direction,
                'confidence': 50.0,
                'predicted_price': predicted_price,
                'model_type': 'Ensemble (Ultimate Fallback)'
            }