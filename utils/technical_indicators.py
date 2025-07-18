import pandas as pd
import numpy as np

class TechnicalIndicators:
    """Technical indicators calculator for stock analysis"""
    
    def __init__(self):
        pass
    
    def check_data_sufficiency(self, data, min_periods=50):
        """Check if data has sufficient periods for meaningful technical analysis"""
        if data is None or data.empty:
            return False, "No data available"
        
        if len(data) < min_periods:
            return False, f"Insufficient data: {len(data)} periods (minimum {min_periods} recommended)"
        
        return True, "Sufficient data"
    
    def calculate_sma(self, data, window=20):
        """Calculate Simple Moving Average"""
        return data['Close'].rolling(window=window).mean()
    
    def calculate_ema(self, data, window=20):
        """Calculate Exponential Moving Average"""
        return data['Close'].ewm(span=window).mean()
    
    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band, sma
    
    def calculate_stochastic(self, data, k_window=14, d_window=3):
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def calculate_williams_r(self, data, window=14):
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=window).max()
        low_min = data['Low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r
    
    def calculate_atr(self, data, window=14):
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def calculate_obv(self, data):
        """Calculate On-Balance Volume"""
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return obv
    
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def calculate_volatility(self, data, window=20):
        """Calculate price volatility (standard deviation of returns)"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std()
        return volatility
    
    def calculate_momentum(self, data, window=10):
        """Calculate price momentum"""
        momentum = data['Close'] - data['Close'].shift(window)
        return momentum
    
    def calculate_roc(self, data, window=12):
        """Calculate Rate of Change"""
        roc = ((data['Close'] - data['Close'].shift(window)) / data['Close'].shift(window)) * 100
        return roc
    
    def calculate_cci(self, data, window=20):
        """Calculate Commodity Channel Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def add_all_indicators(self, data):
        """Add all technical indicators to the dataframe"""
        df = data.copy()
        
        # Check data sufficiency
        is_sufficient, message = self.check_data_sufficiency(data, min_periods=20)
        if not is_sufficient:
            print(f"Warning: {message}. Some indicators may not be reliable.")
        
        try:
            # Moving Averages
            df['SMA_20'] = self.calculate_sma(df, 20)
            df['SMA_50'] = self.calculate_sma(df, 50)
            df['EMA_20'] = self.calculate_ema(df, 20)
            
            # RSI
            df['RSI'] = self.calculate_rsi(df)
            
            # MACD
            macd, macd_signal, macd_histogram = self.calculate_macd(df)
            df['MACD'] = macd
            df['MACD_Signal'] = macd_signal
            df['MACD_Histogram'] = macd_histogram
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(df)
            df['BB_Upper'] = bb_upper
            df['BB_Lower'] = bb_lower
            df['BB_Middle'] = bb_middle
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(df)
            df['Stoch_K'] = stoch_k
            df['Stoch_D'] = stoch_d
            
            # Williams %R
            df['Williams_R'] = self.calculate_williams_r(df)
            
            # ATR
            df['ATR'] = self.calculate_atr(df)
            
            # OBV
            df['OBV'] = self.calculate_obv(df)
            
            # VWAP
            df['VWAP'] = self.calculate_vwap(df)
            
            # Volatility
            df['Volatility'] = self.calculate_volatility(df)
            
            # Momentum
            df['Momentum'] = self.calculate_momentum(df)
            
            # Rate of Change
            df['ROC'] = self.calculate_roc(df)
            
            # CCI
            df['CCI'] = self.calculate_cci(df)
            
            # Additional derived indicators
            df['Price_Above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
            df['Price_Above_SMA50'] = (df['Close'] > df['SMA_50']).astype(int)
            df['SMA20_Above_SMA50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
            
            # Bollinger Band position
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # RSI signals
            df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
            df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
            
            # Volume analysis
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
        
        # Clean up any infinity or NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def get_trading_signals(self, data):
        """Generate trading signals based on technical indicators"""
        signals = {}
        
        latest = data.iloc[-1]
        
        # RSI signals
        if 'RSI' in data.columns:
            rsi = latest['RSI']
            if rsi > 70:
                signals['RSI'] = 'SELL'
            elif rsi < 30:
                signals['RSI'] = 'BUY'
            else:
                signals['RSI'] = 'HOLD'
        
        # MACD signals
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            if latest['MACD'] > latest['MACD_Signal']:
                signals['MACD'] = 'BUY'
            else:
                signals['MACD'] = 'SELL'
        
        # Bollinger Bands signals
        if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
            if latest['Close'] > latest['BB_Upper']:
                signals['Bollinger'] = 'SELL'
            elif latest['Close'] < latest['BB_Lower']:
                signals['Bollinger'] = 'BUY'
            else:
                signals['Bollinger'] = 'HOLD'
        
        # Moving Average signals
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            if latest['SMA_20'] > latest['SMA_50'] and latest['Close'] > latest['SMA_20']:
                signals['MA'] = 'BUY'
            elif latest['SMA_20'] < latest['SMA_50'] and latest['Close'] < latest['SMA_20']:
                signals['MA'] = 'SELL'
            else:
                signals['MA'] = 'HOLD'
        
        return signals
