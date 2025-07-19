#!/usr/bin/env python3
"""
Comprehensive Backtesting Script for StockTrendAI Project
=========================================================

This script provides advanced backtesting capabilities for the Indian stock market 
prediction models with detailed performance metrics and risk analysis.

Features:
- Multiple trading strategies (Buy & Hold, Moving Average, RSI, MACD)
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar ratios)
- Maximum drawdown analysis
- Portfolio simulation with transaction costs
- Benchmarking against NIFTY 50
- Detailed visualization and reporting

Usage:
    python3 backtest_comprehensive.py [--symbol SYMBOL] [--period PERIOD] [--strategy STRATEGY]
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import argparse
import sys
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Technical indicators for trading strategies"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd, signal)
        histogram = macd - signal_line
        return macd, signal_line, histogram

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 0.1% per trade
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals. Override in subclasses."""
        raise NotImplementedError

class BuyHoldStrategy(TradingStrategy):
    """Simple buy and hold strategy"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at the beginning
        signals.iloc[-1] = -1  # Sell at the end
        return signals

class MovingAverageStrategy(TradingStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        short_ma = TechnicalIndicators.sma(data['Close'], self.short_window)
        long_ma = TechnicalIndicators.sma(data['Close'], self.long_window)
        
        # Generate signals based on MA crossover
        position = 0
        for i in range(1, len(data)):
            if pd.notna(short_ma.iloc[i]) and pd.notna(long_ma.iloc[i]):
                if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
                    signals.iloc[i] = 1  # Buy signal
                    position = 1
                elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                    signals.iloc[i] = -1  # Sell signal
                    position = 0
        
        return signals

class RSIStrategy(TradingStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, rsi_window: int = 14, oversold: float = 30, overbought: float = 70, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.rsi_window = rsi_window
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        rsi = TechnicalIndicators.rsi(data['Close'], self.rsi_window)
        
        position = 0
        for i in range(1, len(data)):
            if pd.notna(rsi.iloc[i]):
                if rsi.iloc[i] < self.oversold and position == 0:
                    signals.iloc[i] = 1  # Buy signal (oversold)
                    position = 1
                elif rsi.iloc[i] > self.overbought and position == 1:
                    signals.iloc[i] = -1  # Sell signal (overbought)
                    position = 0
        
        return signals

class MACDStrategy(TradingStrategy):
    """MACD-based momentum strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, initial_capital: float = 100000):
        super().__init__(initial_capital)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        macd, signal_line, _ = TechnicalIndicators.macd(data['Close'], self.fast, self.slow, self.signal)
        
        position = 0
        for i in range(1, len(data)):
            if pd.notna(macd.iloc[i]) and pd.notna(signal_line.iloc[i]):
                if macd.iloc[i] > signal_line.iloc[i] and position == 0:
                    signals.iloc[i] = 1  # Buy signal
                    position = 1
                elif macd.iloc[i] < signal_line.iloc[i] and position == 1:
                    signals.iloc[i] = -1  # Sell signal
                    position = 0
        
        return signals

class PerformanceMetrics:
    """Calculate comprehensive performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return np.inf
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + PerformanceMetrics.calculate_returns(prices)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = abs(PerformanceMetrics.max_drawdown(prices))
        if max_dd == 0:
            return np.inf
        return annual_return / max_dd
    
    @staticmethod
    def var_95(returns: pd.Series) -> float:
        """Calculate 95% Value at Risk"""
        return np.percentile(returns, 5)

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001
    
    def run_backtest(self, data: pd.DataFrame, strategy: TradingStrategy) -> Dict:
        """Run backtest for a given strategy"""
        signals = strategy.generate_signals(data)
        
        # Initialize portfolio
        portfolio = {
            'positions': 0,
            'cash': self.initial_capital,
            'portfolio_value': [],
            'trades': [],
            'returns': []
        }
        
        for i, (date, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            signal = signals.iloc[i]
            
            # Execute trades based on signals
            if signal == 1 and portfolio['positions'] == 0:  # Buy
                shares_to_buy = int(portfolio['cash'] / (current_price * (1 + self.transaction_cost)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    portfolio['cash'] -= cost
                    portfolio['positions'] = shares_to_buy
                    portfolio['trades'].append({
                        'date': date,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost
                    })
            
            elif signal == -1 and portfolio['positions'] > 0:  # Sell
                proceeds = portfolio['positions'] * current_price * (1 - self.transaction_cost)
                portfolio['cash'] += proceeds
                portfolio['trades'].append({
                    'date': date,
                    'action': 'SELL',
                    'shares': portfolio['positions'],
                    'price': current_price,
                    'proceeds': proceeds
                })
                portfolio['positions'] = 0
            
            # Calculate portfolio value
            portfolio_value = portfolio['cash'] + portfolio['positions'] * current_price
            portfolio['portfolio_value'].append(portfolio_value)
        
        # Calculate final metrics
        portfolio_values = pd.Series(portfolio['portfolio_value'], index=data.index)
        returns = PerformanceMetrics.calculate_returns(portfolio_values)
        
        results = {
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': portfolio['trades'],
            'final_value': portfolio['portfolio_value'][-1],
            'total_return': (portfolio['portfolio_value'][-1] / self.initial_capital - 1) * 100,
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(portfolio_values),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns, portfolio_values),
            'var_95': PerformanceMetrics.var_95(returns),
            'num_trades': len(portfolio['trades']),
            'win_rate': 0  # Calculate win rate from trades
        }
        
        # Calculate win rate
        if len(portfolio['trades']) >= 2:
            profitable_trades = 0
            for i in range(0, len(portfolio['trades']), 2):
                if i + 1 < len(portfolio['trades']):
                    buy_price = portfolio['trades'][i]['price']
                    sell_price = portfolio['trades'][i + 1]['price']
                    if sell_price > buy_price:
                        profitable_trades += 1
            results['win_rate'] = profitable_trades / (len(portfolio['trades']) // 2) * 100 if len(portfolio['trades']) >= 2 else 0
        
        return results

class DataFetcher:
    """Fetch and prepare stock data"""
    
    @staticmethod
    def get_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS') and not symbol.startswith('^'):
                symbol = f"{symbol}.NS"
            
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if data.empty:
                print(f"No data found for symbol: {symbol}")
                return None
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    @staticmethod
    def get_benchmark_data(period: str = "2y") -> pd.DataFrame:
        """Fetch NIFTY 50 benchmark data"""
        return DataFetcher.get_stock_data("^NSEI", period)

class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    @staticmethod
    def plot_portfolio_performance(results_dict: Dict, benchmark_data: pd.DataFrame, stock_data: pd.DataFrame):
        """Plot portfolio performance comparison"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value Comparison', 'Daily Returns Distribution', 'Drawdown Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value comparison
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            fig.add_trace(
                go.Scatter(
                    x=results['portfolio_values'].index,
                    y=results['portfolio_values'],
                    name=f"{strategy_name} Portfolio",
                    line=dict(color=colors[i % len(colors)])
                ),
                row=1, col=1
            )
        
        # Add benchmark and buy-hold
        if benchmark_data is not None:
            benchmark_normalized = (benchmark_data['Close'] / benchmark_data['Close'].iloc[0]) * 100000
            fig.add_trace(
                go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized,
                    name="NIFTY 50 Benchmark",
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=1
            )
        
        stock_normalized = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100000
        fig.add_trace(
            go.Scatter(
                x=stock_normalized.index,
                y=stock_normalized,
                name="Buy & Hold",
                line=dict(color='gray', dash='dot')
            ),
            row=1, col=1
        )
        
        # Returns distribution
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            fig.add_trace(
                go.Histogram(
                    x=results['returns'] * 100,
                    name=f"{strategy_name} Returns",
                    opacity=0.7,
                    nbinsx=50
                ),
                row=2, col=1
            )
        
        # Drawdown analysis
        for i, (strategy_name, results) in enumerate(results_dict.items()):
            cumulative = (1 + results['returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    name=f"{strategy_name} Drawdown",
                    fill='tozeroy',
                    fillcolor=f'rgba({colors[i % len(colors)]}, 0.3)'
                ),
                row=3, col=1
            )
        
        fig.update_layout(height=1200, title_text="Comprehensive Backtest Results")
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        return fig
    
    @staticmethod
    def create_performance_table(results_dict: Dict) -> go.Figure:
        """Create performance metrics comparison table"""
        metrics = ['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 
                  'Calmar Ratio', 'VaR 95% (%)', 'Number of Trades', 'Win Rate (%)']
        
        table_data = []
        for strategy_name, results in results_dict.items():
            row = [
                f"{results['total_return']:.2f}",
                f"{results['sharpe_ratio']:.3f}",
                f"{results['sortino_ratio']:.3f}",
                f"{results['max_drawdown'] * 100:.2f}",
                f"{results['calmar_ratio']:.3f}",
                f"{results['var_95'] * 100:.2f}",
                f"{results['num_trades']}",
                f"{results['win_rate']:.1f}"
            ]
            table_data.append(row)
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=['Strategy'] + metrics,
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[list(results_dict.keys())] + list(zip(*table_data)),
                      fill_color='lavender',
                      align='left'))
        ])
        
        fig.update_layout(title="Performance Metrics Comparison")
        return fig

def main():
    """Main function to run comprehensive backtesting"""
    parser = argparse.ArgumentParser(description='Comprehensive Stock Backtesting')
    parser.add_argument('--symbol', type=str, default='RELIANCE', help='Stock symbol (default: RELIANCE)')
    parser.add_argument('--period', type=str, default='2y', help='Data period (default: 2y)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--save-html', action='store_true', help='Save results as HTML file')
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Backtesting for StockTrendAI")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Initial Capital: ₹{args.capital:,.2f}")
    print("=" * 60)
    
    # Fetch data
    print("📊 Fetching stock data...")
    stock_data = DataFetcher.get_stock_data(args.symbol, args.period)
    if stock_data is None:
        print("❌ Failed to fetch stock data. Exiting.")
        return
    
    print("📈 Fetching benchmark data...")
    benchmark_data = DataFetcher.get_benchmark_data(args.period)
    
    # Align data periods
    if benchmark_data is not None:
        common_start = max(stock_data.index.min(), benchmark_data.index.min())
        common_end = min(stock_data.index.max(), benchmark_data.index.max())
        stock_data = stock_data.loc[common_start:common_end]
        benchmark_data = benchmark_data.loc[common_start:common_end]
    
    print(f"✅ Data loaded: {len(stock_data)} trading days")
    
    # Initialize strategies
    strategies = {
        'Buy & Hold': BuyHoldStrategy(args.capital),
        'Moving Average (20/50)': MovingAverageStrategy(20, 50, args.capital),
        'RSI Mean Reversion': RSIStrategy(14, 30, 70, args.capital),
        'MACD Momentum': MACDStrategy(12, 26, 9, args.capital)
    }
    
    # Run backtests
    print("\n🔄 Running backtests...")
    backtest_engine = BacktestEngine(args.capital)
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"  📋 Testing {strategy_name}...")
        try:
            result = backtest_engine.run_backtest(stock_data, strategy)
            results[strategy_name] = result
            print(f"     ✅ Total Return: {result['total_return']:.2f}%")
            print(f"     📊 Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(f"     📉 Max Drawdown: {result['max_drawdown']*100:.2f}%")
        except Exception as e:
            print(f"     ❌ Error: {e}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    performance_fig = BacktestVisualizer.plot_portfolio_performance(results, benchmark_data, stock_data)
    metrics_table = BacktestVisualizer.create_performance_table(results)
    
    # Display results
    print("\n🎯 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    
    for strategy_name, result in results.items():
        print(f"\n📈 {strategy_name}:")
        print(f"   💰 Final Value: ₹{result['final_value']:,.2f}")
        print(f"   📊 Total Return: {result['total_return']:.2f}%")
        print(f"   ⚡ Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        print(f"   🛡️  Sortino Ratio: {result['sortino_ratio']:.3f}")
        print(f"   📉 Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f"   🎯 Calmar Ratio: {result['calmar_ratio']:.3f}")
        print(f"   ⚠️  VaR (95%): {result['var_95']*100:.2f}%")
        print(f"   🔄 Trades: {result['num_trades']}")
        print(f"   🏆 Win Rate: {result['win_rate']:.1f}%")
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    print(f"\n🏆 BEST STRATEGY (by Sharpe Ratio): {best_strategy}")
    print(f"   Sharpe Ratio: {results[best_strategy]['sharpe_ratio']:.3f}")
    print(f"   Total Return: {results[best_strategy]['total_return']:.2f}%")
    
    # Save results if requested
    if args.save_html:
        html_filename = f"backtest_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_filename, 'w') as f:
            f.write(performance_fig.to_html(include_plotlyjs='cdn'))
            f.write(metrics_table.to_html(include_plotlyjs='cdn'))
        print(f"\n💾 Results saved to: {html_filename}")
    
    # Show interactive plots
    print("\n🎨 Displaying interactive charts...")
    performance_fig.show()
    metrics_table.show()
    
    print("\n✅ Backtesting completed successfully!")

if __name__ == "__main__":
    main()