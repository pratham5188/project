#!/usr/bin/env python3
"""
Simple but Comprehensive Backtesting Script for StockTrendAI
============================================================

A focused backtesting tool that provides essential performance metrics
and analysis for Indian stock market prediction strategies.

Features:
- Multiple trading strategies
- Risk-adjusted performance metrics
- Transaction costs simulation
- Benchmarking capabilities
- Detailed reporting

Usage:
    python3 backtest_simple.py [--symbol SYMBOL] [--period PERIOD]
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import argparse
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators for trading strategies"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class PerformanceAnalyzer:
    """Calculate performance metrics and generate reports"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.risk_free_rate = 0.06  # 6% annual risk-free rate
    
    def calculate_metrics(self, portfolio_values: pd.Series, trades: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Basic returns calculation
        returns = portfolio_values.pct_change().dropna()
        total_return = (portfolio_values.iloc[-1] / self.initial_capital - 1) * 100
        
        # Risk metrics
        sharpe_ratio = self._sharpe_ratio(returns)
        max_drawdown = self._max_drawdown(portfolio_values)
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Trade analysis
        win_rate = self._calculate_win_rate(trades)
        
        return {
            'final_value': portfolio_values.iloc[-1],
            'total_return': total_return,
            'annualized_return': (1 + total_return/100) ** (252/len(portfolio_values)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'best_day': returns.max() * 100,
            'worst_day': returns.min() * 100
        }
    
    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        returns = prices.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if len(trades) < 2:
            return 0
        
        profitable_trades = 0
        total_trades = 0
        
        i = 0
        while i < len(trades) - 1:
            if trades[i]['action'] == 'BUY' and trades[i+1]['action'] == 'SELL':
                if trades[i+1]['price'] > trades[i]['price']:
                    profitable_trades += 1
                total_trades += 1
                i += 2
            else:
                i += 1
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0

class BacktestStrategy:
    """Base class for backtesting strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 0.1% per trade
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest and return results"""
        signals = self.generate_signals(data)
        
        # Initialize portfolio
        cash = self.initial_capital
        positions = 0
        portfolio_values = []
        trades = []
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            signal = signals.iloc[i] if i < len(signals) else 0
            
            # Execute trades
            if signal == 1 and positions == 0:  # Buy signal
                shares_to_buy = int(cash / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    cash -= cost
                    positions = shares_to_buy
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
            
            elif signal == -1 and positions > 0:  # Sell signal
                proceeds = positions * current_price * (1 - self.transaction_cost)
                cash += proceeds
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'shares': positions,
                    'price': current_price,
                    'proceeds': proceeds
                })
                positions = 0
            
            # Calculate portfolio value
            portfolio_value = cash + positions * current_price
            portfolio_values.append(portfolio_value)
        
        portfolio_values = pd.Series(portfolio_values, index=data.index)
        
        # Calculate performance metrics
        analyzer = PerformanceAnalyzer(self.initial_capital)
        metrics = analyzer.calculate_metrics(portfolio_values, trades)
        
        return {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'metrics': metrics
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals - override in subclasses"""
        raise NotImplementedError

class BuyHoldStrategy(BacktestStrategy):
    """Simple buy and hold strategy"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = -1  # Sell at end
        return signals

class MovingAverageStrategy(BacktestStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        short_ma = TechnicalIndicators.sma(data['Close'], self.short_window)
        long_ma = TechnicalIndicators.sma(data['Close'], self.long_window)
        
        position = 0
        for i in range(1, len(data)):
            if pd.notna(short_ma.iloc[i]) and pd.notna(long_ma.iloc[i]):
                if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
                    signals.iloc[i] = 1  # Buy
                    position = 1
                elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                    signals.iloc[i] = -1  # Sell
                    position = 0
        
        return signals

class RSIStrategy(BacktestStrategy):
    """RSI mean reversion strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        rsi = TechnicalIndicators.rsi(data['Close'], self.rsi_period)
        
        position = 0
        for i in range(1, len(data)):
            if pd.notna(rsi.iloc[i]):
                if rsi.iloc[i] < self.oversold and position == 0:
                    signals.iloc[i] = 1  # Buy (oversold)
                    position = 1
                elif rsi.iloc[i] > self.overbought and position == 1:
                    signals.iloc[i] = -1  # Sell (overbought)
                    position = 0
        
        return signals

def fetch_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance"""
    try:
        # Add .NS suffix for Indian stocks
        if not symbol.endswith('.NS') and not symbol.startswith('^'):
            symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"❌ No data found for symbol: {symbol}")
            return None
        
        return data
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

def print_results_table(results: Dict[str, Dict]):
    """Print a formatted results table"""
    print("\n" + "="*100)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"{'Strategy':<25} {'Return %':<12} {'Sharpe':<10} {'Max DD %':<12} {'Volatility %':<15} {'Trades':<10} {'Win Rate %':<12}")
    print("-"*100)
    
    # Results
    for strategy_name, result in results.items():
        metrics = result['metrics']
        print(f"{strategy_name:<25} "
              f"{metrics['total_return']:>8.2f}%   "
              f"{metrics['sharpe_ratio']:>7.3f}   "
              f"{metrics['max_drawdown']:>8.2f}%   "
              f"{metrics['volatility']:>11.2f}%   "
              f"{metrics['num_trades']:>6}     "
              f"{metrics['win_rate']:>8.1f}%")

def print_detailed_analysis(strategy_name: str, result: Dict):
    """Print detailed analysis for a strategy"""
    metrics = result['metrics']
    
    print(f"\n📈 DETAILED ANALYSIS: {strategy_name}")
    print("-" * 50)
    print(f"💰 Initial Capital:     ₹{100000:>10,.2f}")
    print(f"💰 Final Value:         ₹{metrics['final_value']:>10,.2f}")
    print(f"📊 Total Return:        {metrics['total_return']:>10.2f}%")
    print(f"📊 Annualized Return:   {metrics['annualized_return']*100:>10.2f}%")
    print(f"⚡ Sharpe Ratio:        {metrics['sharpe_ratio']:>10.3f}")
    print(f"📉 Max Drawdown:        {metrics['max_drawdown']:>10.2f}%")
    print(f"📊 Volatility:          {metrics['volatility']:>10.2f}%")
    print(f"🔄 Number of Trades:    {metrics['num_trades']:>10}")
    print(f"🏆 Win Rate:            {metrics['win_rate']:>10.1f}%")
    print(f"📈 Best Day:            {metrics['best_day']:>10.2f}%")
    print(f"📉 Worst Day:           {metrics['worst_day']:>10.2f}%")

def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description='Stock Backtesting for StockTrendAI')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol (default: RELIANCE)')
    parser.add_argument('--period', type=str, default='2y', help='Data period (default: 2y)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis for each strategy')
    
    args = parser.parse_args()
    
    print("🚀 StockTrendAI Backtesting Engine")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Initial Capital: ₹{args.capital:,.2f}")
    print("=" * 60)
    
    # Fetch data
    print("📊 Fetching stock data...")
    stock_data = fetch_stock_data(args.symbol, args.period)
    if stock_data is None:
        return
    
    print(f"✅ Data loaded: {len(stock_data)} trading days")
    print(f"📅 Period: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Initialize strategies
    strategies = {
        'Buy & Hold': BuyHoldStrategy(args.capital),
        'MA Crossover (20/50)': MovingAverageStrategy(20, 50, args.capital),
        'MA Crossover (10/30)': MovingAverageStrategy(10, 30, args.capital),
        'RSI Mean Reversion': RSIStrategy(14, 30, 70, args.capital),
        'RSI Aggressive': RSIStrategy(14, 25, 75, args.capital)
    }
    
    # Run backtests
    print("\n🔄 Running backtests...")
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"  📋 Testing {strategy_name}...")
        try:
            result = strategy.run_backtest(stock_data)
            results[strategy_name] = result
            metrics = result['metrics']
            print(f"     ✅ Return: {metrics['total_return']:>8.2f}% | "
                  f"Sharpe: {metrics['sharpe_ratio']:>6.3f} | "
                  f"Trades: {metrics['num_trades']:>3}")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Print results
    print_results_table(results)
    
    # Find and highlight best strategy
    if results:
        best_strategy = max(results.keys(), key=lambda x: results[x]['metrics']['sharpe_ratio'])
        best_return_strategy = max(results.keys(), key=lambda x: results[x]['metrics']['total_return'])
        
        print(f"\n🏆 BEST STRATEGIES:")
        print(f"📊 Highest Sharpe Ratio: {best_strategy} ({results[best_strategy]['metrics']['sharpe_ratio']:.3f})")
        print(f"💰 Highest Return:       {best_return_strategy} ({results[best_return_strategy]['metrics']['total_return']:.2f}%)")
        
        # Show detailed analysis if requested
        if args.detailed:
            for strategy_name, result in results.items():
                print_detailed_analysis(strategy_name, result)
    
    # Simple benchmark comparison
    buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
    print(f"\n📈 BENCHMARK COMPARISON:")
    print(f"Simple Buy & Hold (no costs): {buy_hold_return:.2f}%")
    
    print("\n✅ Backtesting completed successfully!")

if __name__ == "__main__":
    main()