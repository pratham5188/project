#!/usr/bin/env python3
"""
Advanced Trading Strategies for StockTrendAI
============================================

This module contains advanced trading strategies that can be used
with the backtesting framework.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from backtest_simple import BacktestStrategy, TechnicalIndicators

class BollingerBandsStrategy(BacktestStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, window=20, num_std=2, initial_capital=100000):
        super().__init__(initial_capital)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=self.window).mean()
        std = data['Close'].rolling(window=self.window).std()
        upper_band = sma + (std * self.num_std)
        lower_band = sma - (std * self.num_std)
        
        position = 0
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            
            if pd.notna(upper_band.iloc[i]) and pd.notna(lower_band.iloc[i]):
                # Buy when price touches lower band (oversold)
                if current_price <= lower_band.iloc[i] and position == 0:
                    signals.iloc[i] = 1
                    position = 1
                # Sell when price touches upper band (overbought)
                elif current_price >= upper_band.iloc[i] and position == 1:
                    signals.iloc[i] = -1
                    position = 0
                # Exit when price returns to middle (SMA)
                elif position == 1 and current_price >= sma.iloc[i]:
                    signals.iloc[i] = -1
                    position = 0
        
        return signals

class MomentumStrategy(BacktestStrategy):
    """Price momentum strategy"""
    
    def __init__(self, lookback=10, threshold=0.02, initial_capital=100000):
        super().__init__(initial_capital)
        self.lookback = lookback
        self.threshold = threshold  # 2% momentum threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        # Calculate momentum (rate of change)
        momentum = data['Close'].pct_change(self.lookback)
        
        position = 0
        for i in range(self.lookback + 1, len(data)):
            current_momentum = momentum.iloc[i]
            
            if pd.notna(current_momentum):
                # Buy on strong positive momentum
                if current_momentum > self.threshold and position == 0:
                    signals.iloc[i] = 1
                    position = 1
                # Sell on negative momentum or momentum weakening
                elif current_momentum < -self.threshold/2 and position == 1:
                    signals.iloc[i] = -1
                    position = 0
        
        return signals

class VolatilityBreakoutStrategy(BacktestStrategy):
    """Volatility breakout strategy using ATR"""
    
    def __init__(self, atr_period=14, breakout_multiplier=2.0, initial_capital=100000):
        super().__init__(initial_capital)
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        atr = self.calculate_atr(data)
        
        position = 0
        entry_price = 0
        
        for i in range(self.atr_period + 1, len(data)):
            current_price = data['Close'].iloc[i]
            prev_close = data['Close'].iloc[i-1]
            current_atr = atr.iloc[i]
            
            if pd.notna(current_atr):
                # Buy on upward breakout
                if current_price > prev_close + (current_atr * self.breakout_multiplier) and position == 0:
                    signals.iloc[i] = 1
                    position = 1
                    entry_price = current_price
                
                # Sell on downward breakout or stop loss
                elif position == 1:
                    if (current_price < prev_close - (current_atr * self.breakout_multiplier) or 
                        current_price < entry_price - (current_atr * self.breakout_multiplier)):
                        signals.iloc[i] = -1
                        position = 0
        
        return signals

class DualTimeframeStrategy(BacktestStrategy):
    """Dual timeframe strategy using multiple moving averages"""
    
    def __init__(self, fast_ma=5, slow_ma=20, trend_ma=50, initial_capital=100000):
        super().__init__(initial_capital)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.trend_ma = trend_ma
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving averages
        fast_ma = TechnicalIndicators.sma(data['Close'], self.fast_ma)
        slow_ma = TechnicalIndicators.sma(data['Close'], self.slow_ma)
        trend_ma = TechnicalIndicators.sma(data['Close'], self.trend_ma)
        
        position = 0
        for i in range(self.trend_ma + 1, len(data)):
            current_price = data['Close'].iloc[i]
            
            if (pd.notna(fast_ma.iloc[i]) and pd.notna(slow_ma.iloc[i]) and 
                pd.notna(trend_ma.iloc[i])):
                
                # Only trade in direction of long-term trend
                is_uptrend = current_price > trend_ma.iloc[i]
                
                # Buy signal: fast MA crosses above slow MA in uptrend
                if (fast_ma.iloc[i] > slow_ma.iloc[i] and 
                    fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and 
                    is_uptrend and position == 0):
                    signals.iloc[i] = 1
                    position = 1
                
                # Sell signal: fast MA crosses below slow MA or trend reversal
                elif (position == 1 and 
                      (fast_ma.iloc[i] < slow_ma.iloc[i] or not is_uptrend)):
                    signals.iloc[i] = -1
                    position = 0
        
        return signals

class MeanReversionStrategy(BacktestStrategy):
    """Advanced mean reversion strategy with multiple indicators"""
    
    def __init__(self, rsi_period=14, bb_period=20, bb_std=2, initial_capital=100000):
        super().__init__(initial_capital)
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        # Calculate indicators
        rsi = TechnicalIndicators.rsi(data['Close'], self.rsi_period)
        sma = data['Close'].rolling(window=self.bb_period).mean()
        std = data['Close'].rolling(window=self.bb_period).std()
        lower_band = sma - (std * self.bb_std)
        upper_band = sma + (std * self.bb_std)
        
        position = 0
        for i in range(max(self.rsi_period, self.bb_period) + 1, len(data)):
            current_price = data['Close'].iloc[i]
            current_rsi = rsi.iloc[i]
            
            if (pd.notna(current_rsi) and pd.notna(lower_band.iloc[i]) and 
                pd.notna(upper_band.iloc[i])):
                
                # Strong buy signal: both RSI oversold AND price below lower BB
                if (current_rsi < 30 and current_price < lower_band.iloc[i] and position == 0):
                    signals.iloc[i] = 1
                    position = 1
                
                # Strong sell signal: both RSI overbought AND price above upper BB
                elif (current_rsi > 70 and current_price > upper_band.iloc[i] and position == 1):
                    signals.iloc[i] = -1
                    position = 0
                
                # Exit on return to neutral: RSI back to middle range
                elif (position == 1 and 40 < current_rsi < 60):
                    signals.iloc[i] = -1
                    position = 0
        
        return signals

class TrendFollowingStrategy(BacktestStrategy):
    """Advanced trend following with multiple confirmations"""
    
    def __init__(self, ema_fast=12, ema_slow=26, atr_period=14, initial_capital=100000):
        super().__init__(initial_capital)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        # Calculate indicators
        ema_fast = TechnicalIndicators.ema(data['Close'], self.ema_fast)
        ema_slow = TechnicalIndicators.ema(data['Close'], self.ema_slow)
        atr = self.calculate_atr(data)
        
        position = 0
        entry_price = 0
        
        for i in range(max(self.ema_slow, self.atr_period) + 1, len(data)):
            current_price = data['Close'].iloc[i]
            current_atr = atr.iloc[i]
            
            if (pd.notna(ema_fast.iloc[i]) and pd.notna(ema_slow.iloc[i]) and 
                pd.notna(current_atr)):
                
                # Trend strength: difference between EMAs relative to ATR
                ema_diff = ema_fast.iloc[i] - ema_slow.iloc[i]
                trend_strength = abs(ema_diff) / current_atr
                
                # Buy signal: fast EMA above slow EMA with strong trend
                if (ema_fast.iloc[i] > ema_slow.iloc[i] and 
                    ema_fast.iloc[i-1] <= ema_slow.iloc[i-1] and 
                    trend_strength > 0.5 and position == 0):
                    signals.iloc[i] = 1
                    position = 1
                    entry_price = current_price
                
                # Sell signal: trend reversal or stop loss
                elif position == 1:
                    if (ema_fast.iloc[i] < ema_slow.iloc[i] or 
                        current_price < entry_price - (2 * current_atr)):
                        signals.iloc[i] = -1
                        position = 0
        
        return signals

# Strategy registry for easy access
ADVANCED_STRATEGIES = {
    'Bollinger Bands': BollingerBandsStrategy,
    'Momentum': MomentumStrategy,
    'Volatility Breakout': VolatilityBreakoutStrategy,
    'Dual Timeframe': DualTimeframeStrategy,
    'Mean Reversion Plus': MeanReversionStrategy,
    'Trend Following Plus': TrendFollowingStrategy
}

def get_strategy(strategy_name: str, **kwargs):
    """Get strategy instance by name"""
    if strategy_name in ADVANCED_STRATEGIES:
        return ADVANCED_STRATEGIES[strategy_name](**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def list_strategies():
    """List all available strategies"""
    return list(ADVANCED_STRATEGIES.keys())