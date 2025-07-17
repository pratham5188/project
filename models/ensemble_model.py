import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class EnsemblePredictor:
    """Ensemble model combining multiple ML algorithms for robust predictions"""
    
    def __init__(self):
        self.classification_ensemble = None
        self.regression_ensemble = None
        self.scaler = StandardScaler()
        self.sklearn_available = SKLEARN_AVAILABLE
        self.feature_columns = []
        
    def prepare_features(self, data):
        """Prepare comprehensive feature set"""
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['open'] = data['Open']
        features['high'] = data['High']
        features['low'] = data['Low']
        features['volume'] = data['Volume']
        features['close'] = data['Close']
        
        # Technical indicators
        if 'SMA_20' in data.columns:
            features['sma_20'] = data['SMA_20']
        if 'SMA_50' in data.columns:
            features['sma_50'] = data['SMA_50']
        if 'RSI' in data.columns:
            features['rsi'] = data['RSI']
        if 'MACD' in data.columns:
            features['macd'] = data['MACD']
        if 'MACD_Signal' in data.columns:
            features['macd_signal'] = data['MACD_Signal']
        
        # Derived features
        features['price_change'] = data['Close'].pct_change()
        features['volume_change'] = data['Volume'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Moving averages of ratios
        features['ma5_close'] = data['Close'].rolling(5).mean()
        features['ma10_close'] = data['Close'].rolling(10).mean()
        features['ma20_close'] = data['Close'].rolling(20).mean()
        
        # Volatility measures
        features['volatility_5'] = data['Close'].rolling(5).std()
        features['volatility_20'] = data['Close'].rolling(20).std()
        
        # Price position relative to recent highs/lows
        features['price_vs_5day_high'] = data['Close'] / data['High'].rolling(5).max()
        features['price_vs_5day_low'] = data['Close'] / data['Low'].rolling(5).min()
        features['price_vs_20day_high'] = data['Close'] / data['High'].rolling(20).max()
        features['price_vs_20day_low'] = data['Close'] / data['Low'].rolling(20).min()
        
        # Momentum indicators
        features['momentum_5'] = data['Close'] / data['Close'].shift(5)
        features['momentum_10'] = data['Close'] / data['Close'].shift(10)
        
        # Volume indicators
        features['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        return features.fillna(0)
    
    def create_targets(self, data):
        """Create both classification and regression targets"""
        # Classification target: 1 for up, 0 for down
        price_change = data['Close'].pct_change().shift(-1)
        classification_target = (price_change > 0).astype(int)
        
        # Regression target: actual next day price
        regression_target = data['Close'].shift(-1)
        
        return classification_target, regression_target
    
    def build_ensemble_models(self):
        """Build ensemble models for both classification and regression"""
        if not self.sklearn_available:
            return False
        
        # Classification ensemble (direction prediction)
        clf_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ]
        
        # Add SVM if we have enough data
        try:
            clf_models.append(('svm', SVC(probability=True, random_state=42)))
        except:
            pass
        
        self.classification_ensemble = VotingClassifier(
            estimators=clf_models,
            voting='soft'
        )
        
        # Regression ensemble (price prediction)
        reg_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('lr', LinearRegression()),
        ]
        
        # Add SVR if available
        try:
            reg_models.append(('svr', SVR(kernel='rbf')))
        except:
            pass
        
        self.regression_ensemble = VotingRegressor(estimators=reg_models)
        
        return True
    
    def train(self, data):
        """Train the ensemble models"""
        try:
            if not self.sklearn_available:
                print("Scikit-learn not available for ensemble training")
                return None
            
            # Prepare features and targets
            features = self.prepare_features(data.copy())
            clf_target, reg_target = self.create_targets(data.copy())
            
            # Remove rows with NaN targets
            valid_mask = ~(clf_target.isna() | reg_target.isna())
            features = features[valid_mask]
            clf_target = clf_target[valid_mask]
            reg_target = reg_target[valid_mask]
            
            if len(features) < 50:
                raise ValueError("Insufficient data for ensemble training")
            
            # Store feature columns
            self.feature_columns = features.columns.tolist()
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Build and train models
            if not self.build_ensemble_models():
                return None
            
            # Train classification ensemble
            self.classification_ensemble.fit(features_scaled, clf_target)
            
            # Train regression ensemble
            self.regression_ensemble.fit(features_scaled, reg_target)
            
            # Calculate cross-validation scores
            clf_cv_score = cross_val_score(self.classification_ensemble, features_scaled, clf_target, cv=5).mean()
            reg_cv_score = cross_val_score(self.regression_ensemble, features_scaled, reg_target, cv=5).mean()
            
            return {
                'classification_accuracy': clf_cv_score,
                'regression_score': reg_cv_score
            }
            
        except Exception as e:
            print(f"Ensemble training error: {str(e)}")
            return None
    
    def advanced_technical_prediction(self, data):
        """Advanced technical analysis based prediction"""
        current_price = data['Close'].iloc[-1]
        
        # Multiple timeframe analysis
        short_sma = data['Close'].rolling(5).mean().iloc[-1]
        medium_sma = data['Close'].rolling(20).mean().iloc[-1]
        long_sma = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else medium_sma
        
        # RSI analysis
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        
        # MACD analysis
        macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else 0
        macd_signal = data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in data.columns else 0
        
        # Volume analysis
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Scoring system
        score = 0
        
        # Trend signals
        if current_price > short_sma > medium_sma > long_sma:
            score += 3  # Strong uptrend
        elif current_price > short_sma > medium_sma:
            score += 2  # Medium uptrend
        elif current_price > short_sma:
            score += 1  # Weak uptrend
        elif current_price < short_sma < medium_sma < long_sma:
            score -= 3  # Strong downtrend
        elif current_price < short_sma < medium_sma:
            score -= 2  # Medium downtrend
        elif current_price < short_sma:
            score -= 1  # Weak downtrend
        
        # RSI signals
        if rsi < 30:
            score += 1  # Oversold, potential upward move
        elif rsi > 70:
            score -= 1  # Overbought, potential downward move
        
        # MACD signals
        if macd > macd_signal:
            score += 1  # Bullish crossover
        else:
            score -= 1  # Bearish crossover
        
        # Volume confirmation
        if volume_ratio > 1.5:  # High volume
            score = score * 1.2 if score > 0 else score * 1.2
        
        # Convert score to prediction
        if score >= 2:
            direction = 'UP'
            confidence = min(80.0, 60.0 + abs(score) * 3)
            predicted_price = current_price * (1 + 0.01 * abs(score))
        elif score <= -2:
            direction = 'DOWN'
            confidence = min(80.0, 60.0 + abs(score) * 3)
            predicted_price = current_price * (1 - 0.01 * abs(score))
        else:
            direction = 'UP' if current_price > data['Close'].iloc[-2] else 'DOWN'
            confidence = 55.0
            predicted_price = current_price * (1.005 if direction == 'UP' else 0.995)
        
        return {
            'direction': direction,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'model_type': 'Ensemble (Technical Analysis)',
            'technical_score': score
        }
    
    def predict(self, data):
        """Make ensemble prediction"""
        try:
            if not self.sklearn_available:
                return self.advanced_technical_prediction(data)
            
            # Train models if not available
            if self.classification_ensemble is None or self.regression_ensemble is None:
                print("Training new ensemble models...")
                result = self.train(data.copy())
                if result is None:
                    return self.advanced_technical_prediction(data)
            
            # Prepare features for latest data point
            features = self.prepare_features(data.copy())
            latest_features = features.iloc[-1:][self.feature_columns]
            
            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Get predictions
            direction_prob = self.classification_ensemble.predict_proba(latest_features_scaled)[0]
            predicted_price = self.regression_ensemble.predict(latest_features_scaled)[0]
            
            # Determine direction and confidence
            up_probability = direction_prob[1] if len(direction_prob) > 1 else 0.5
            direction = 'UP' if up_probability > 0.5 else 'DOWN'
            confidence = max(abs(up_probability - 0.5) * 200, 50.0)
            confidence = min(confidence, 85.0)
            
            # Ensure reasonable price prediction
            current_price = data['Close'].iloc[-1]
            max_change = 0.05  # 5% max change
            if abs(predicted_price - current_price) / current_price > max_change:
                if predicted_price > current_price:
                    predicted_price = current_price * (1 + max_change)
                else:
                    predicted_price = current_price * (1 - max_change)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'model_type': 'Ensemble (ML Models)',
                'up_probability': up_probability,
                'model_components': len(self.classification_ensemble.estimators_)
            }
            
        except Exception as e:
            print(f"Ensemble prediction error: {str(e)}")
            return self.advanced_technical_prediction(data)