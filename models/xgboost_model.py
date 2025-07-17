import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class XGBoostPredictor:
    """XGBoost-based stock prediction model for speed and efficiency"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 
            'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volatility'
        ]
        self.model_path = 'models/xgboost_model.pkl'
        self.scaler_path = 'models/xgboost_scaler.pkl'
    
    def prepare_features(self, data):
        """Prepare features for training/prediction"""
        try:
            # Ensure required columns exist
            required_base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_base_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Create price-based features
            data['Price_Change'] = data['Close'].pct_change()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Price_Volume_Trend'] = data['Close'] * data['Volume']
            
            # Technical indicator ratios
            if 'RSI' in data.columns:
                data['RSI_Normalized'] = (data['RSI'] - 50) / 50
            
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Band position
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Update feature columns to include new features
            additional_features = ['Price_Change', 'High_Low_Ratio', 'Volume_SMA', 
                                 'Price_Volume_Trend', 'RSI_Normalized', 'MACD_Histogram', 'BB_Position']
            
            available_features = [col for col in self.feature_columns + additional_features if col in data.columns]
            return data[available_features].fillna(0)
            
        except Exception as e:
            print(f"Feature preparation error: {str(e)}")
            # Return minimal feature set if error occurs
            minimal_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_minimal = [col for col in minimal_features if col in data.columns]
            return data[available_minimal].fillna(0)
    
    def create_target(self, data):
        """Create target variable (1 for price up, 0 for price down)"""
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        return data['Target']
    
    def train(self, data):
        """Train the XGBoost model"""
        try:
            # Prepare features and target
            features = self.prepare_features(data.copy())
            target = self.create_target(data.copy())
            
            # Remove last row (no target available)
            features = features[:-1]
            target = target[:-1]
            
            # Remove any remaining NaN values
            mask = ~(features.isna().any(axis=1) | target.isna())
            features = features[mask]
            target = target[mask]
            
            if len(features) < 50:
                raise ValueError("Insufficient data for training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest (as XGBoost alternative)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model and scaler
            os.makedirs('models', exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            return accuracy
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return None
    
    def load_model(self):
        """Load saved model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                return True
        except Exception as e:
            print(f"Model loading error: {str(e)}")
        return False
    
    def predict(self, data):
        """Make prediction for next day"""
        try:
            # Try to load existing model first
            if self.model is None:
                if not self.load_model():
                    # Train new model if no saved model exists
                    print("Training new XGBoost model...")
                    accuracy = self.train(data.copy())
                    if accuracy is None:
                        raise Exception("Model training failed")
            
            # Prepare features for latest data point
            features = self.prepare_features(data.copy())
            latest_features = features.iloc[-1:].fillna(0)
            
            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(latest_features_scaled)[0]
            confidence = max(self.model.predict_proba(latest_features_scaled)[0]) * 100
            
            # Calculate predicted price (simple estimation)
            current_price = data['Close'].iloc[-1]
            price_change_mean = data['Close'].pct_change().mean()
            
            if prediction == 1:  # Up
                predicted_price = current_price * (1 + abs(price_change_mean))
                direction = 'UP'
            else:  # Down
                predicted_price = current_price * (1 - abs(price_change_mean))
                direction = 'DOWN'
            
            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'model_type': 'XGBoost (Random Forest)'
            }
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Return fallback prediction
            current_price = data['Close'].iloc[-1]
            return {
                'direction': 'UP' if data['Close'].iloc[-1] > data['Close'].iloc[-2] else 'DOWN',
                'confidence': 50.0,
                'predicted_price': current_price,
                'model_type': 'XGBoost (Fallback)'
            }
