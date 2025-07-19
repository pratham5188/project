#!/usr/bin/env python3
"""
Multi-Stock Backtesting Analysis for StockTrendAI
=================================================

This script tests multiple Indian stocks across different sectors to identify
which strategies perform best across various market conditions and stock types.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting modules
sys.path.append('.')
from backtest_simple import BuyHoldStrategy, MovingAverageStrategy, RSIStrategy, fetch_stock_data

class MultiStockAnalyzer:
    """Analyze multiple stocks with different strategies"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.results = {}
        
        # Define stock universe by sectors
        self.stock_universe = {
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'Consumer': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
            'Energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL'],
            'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LUPIN'],
            'Telecom': ['BHARTIARTL', 'IDEA', 'MTNL'],
            'Steel': ['TATASTEEL', 'JSWSTEEL', 'SAIL', 'HINDALCO']
        }
        
        # Define strategies with different parameters
        self.strategies = {
            'Buy & Hold': BuyHoldStrategy(initial_capital),
            'MA Fast (5/15)': MovingAverageStrategy(5, 15, initial_capital),
            'MA Medium (10/30)': MovingAverageStrategy(10, 30, initial_capital),
            'MA Slow (20/50)': MovingAverageStrategy(20, 50, initial_capital),
            'RSI Conservative': RSIStrategy(14, 20, 80, initial_capital),
            'RSI Standard': RSIStrategy(14, 30, 70, initial_capital),
            'RSI Aggressive': RSIStrategy(14, 40, 60, initial_capital)
        }
    
    def test_single_stock(self, symbol, period='1y'):
        """Test all strategies on a single stock"""
        print(f"  📊 Testing {symbol}...")
        
        # Fetch data
        stock_data = fetch_stock_data(symbol, period)
        if stock_data is None or len(stock_data) < 50:
            print(f"    ❌ Insufficient data for {symbol}")
            return None
        
        stock_results = {}
        
        # Test each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                result = strategy.run_backtest(stock_data)
                stock_results[strategy_name] = result['metrics']
            except Exception as e:
                print(f"    ⚠️  Error testing {strategy_name} on {symbol}: {e}")
                stock_results[strategy_name] = None
        
        # Calculate simple buy-hold benchmark
        benchmark_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
        stock_results['Benchmark'] = {'total_return': benchmark_return}
        
        return stock_results
    
    def test_sector(self, sector_name, period='1y'):
        """Test all stocks in a sector"""
        print(f"\n🔍 Testing {sector_name} Sector:")
        print("-" * 40)
        
        sector_stocks = self.stock_universe.get(sector_name, [])
        sector_results = {}
        
        for stock in sector_stocks:
            result = self.test_single_stock(stock, period)
            if result:
                sector_results[stock] = result
        
        return sector_results
    
    def test_all_stocks(self, period='1y'):
        """Test all stocks across all sectors"""
        print(f"🚀 Multi-Stock Backtesting Analysis")
        print(f"Period: {period}")
        print("=" * 60)
        
        all_results = {}
        
        for sector_name in self.stock_universe.keys():
            sector_results = self.test_sector(sector_name, period)
            all_results[sector_name] = sector_results
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """Analyze and summarize results across all stocks"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print(f"\n📈 MULTI-STOCK ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Collect all strategy performances
        strategy_performances = {}
        for strategy_name in self.strategies.keys():
            strategy_performances[strategy_name] = {
                'returns': [],
                'sharpe_ratios': [],
                'max_drawdowns': [],
                'win_rates': [],
                'positive_returns': 0,
                'total_tests': 0
            }
        
        # Aggregate data
        for sector_name, sector_data in self.results.items():
            for stock, stock_results in sector_data.items():
                for strategy_name, metrics in stock_results.items():
                    if strategy_name in strategy_performances and metrics:
                        perf = strategy_performances[strategy_name]
                        perf['returns'].append(metrics.get('total_return', 0))
                        perf['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))
                        perf['max_drawdowns'].append(metrics.get('max_drawdown', 0))
                        perf['win_rates'].append(metrics.get('win_rate', 0))
                        perf['total_tests'] += 1
                        if metrics.get('total_return', 0) > 0:
                            perf['positive_returns'] += 1
        
        # Calculate summary statistics
        print(f"{'Strategy':<20} {'Avg Return':<12} {'Avg Sharpe':<12} {'Success Rate':<12} {'Avg Max DD':<12}")
        print("-" * 80)
        
        strategy_rankings = []
        
        for strategy_name, perf in strategy_performances.items():
            if perf['total_tests'] > 0:
                avg_return = np.mean(perf['returns'])
                avg_sharpe = np.mean(perf['sharpe_ratios'])
                success_rate = (perf['positive_returns'] / perf['total_tests']) * 100
                avg_max_dd = np.mean(perf['max_drawdowns'])
                
                strategy_rankings.append({
                    'strategy': strategy_name,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'success_rate': success_rate,
                    'avg_max_dd': avg_max_dd,
                    'total_tests': perf['total_tests']
                })
                
                print(f"{strategy_name:<20} {avg_return:>8.2f}%    {avg_sharpe:>8.3f}    "
                      f"{success_rate:>8.1f}%     {avg_max_dd:>8.2f}%")
        
        # Find best strategies
        print(f"\n🏆 STRATEGY RANKINGS:")
        print("-" * 40)
        
        # Best by average return
        best_return = max(strategy_rankings, key=lambda x: x['avg_return'])
        print(f"💰 Best Average Return: {best_return['strategy']} ({best_return['avg_return']:.2f}%)")
        
        # Best by Sharpe ratio
        best_sharpe = max(strategy_rankings, key=lambda x: x['avg_sharpe'])
        print(f"⚡ Best Risk-Adjusted: {best_sharpe['strategy']} (Sharpe: {best_sharpe['avg_sharpe']:.3f})")
        
        # Best success rate
        best_success = max(strategy_rankings, key=lambda x: x['success_rate'])
        print(f"🎯 Most Consistent: {best_success['strategy']} ({best_success['success_rate']:.1f}% success)")
        
        # Sector analysis
        self.analyze_sectors()
        
        return strategy_rankings
    
    def analyze_sectors(self):
        """Analyze performance by sector"""
        print(f"\n📊 SECTOR ANALYSIS:")
        print("-" * 50)
        
        sector_performance = {}
        
        for sector_name, sector_data in self.results.items():
            if not sector_data:
                continue
                
            sector_returns = []
            for stock, stock_results in sector_data.items():
                # Use the best performing strategy for each stock
                best_return = -100
                for strategy_name, metrics in stock_results.items():
                    if strategy_name != 'Benchmark' and metrics:
                        if metrics.get('total_return', -100) > best_return:
                            best_return = metrics.get('total_return', -100)
                
                if best_return > -100:
                    sector_returns.append(best_return)
            
            if sector_returns:
                avg_sector_return = np.mean(sector_returns)
                sector_performance[sector_name] = avg_sector_return
                print(f"{sector_name:<15} Average Best Return: {avg_sector_return:>8.2f}%")
        
        # Best performing sector
        if sector_performance:
            best_sector = max(sector_performance.items(), key=lambda x: x[1])
            print(f"\n🏆 Best Performing Sector: {best_sector[0]} ({best_sector[1]:.2f}%)")

def main():
    """Main function to run multi-stock analysis"""
    
    # Check available periods
    periods = {
        '1': '6mo',
        '2': '1y',
        '3': '2y'
    }
    
    print("🚀 StockTrendAI Multi-Stock Backtesting")
    print("=" * 50)
    print("Select analysis period:")
    print("1. 6 months (faster)")
    print("2. 1 year (recommended)")
    print("3. 2 years (comprehensive)")
    
    choice = input("\nEnter choice (1-3, default: 2): ").strip() or '2'
    period = periods.get(choice, '1y')
    
    print("Select analysis scope:")
    print("1. Quick test (5 stocks)")
    print("2. Sector analysis (choose sector)")
    print("3. Full analysis (all stocks)")
    
    scope_choice = input("Enter choice (1-3, default: 1): ").strip() or '1'
    
    analyzer = MultiStockAnalyzer()
    
    if scope_choice == '1':
        # Quick test with popular stocks
        test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
        print(f"\n🔄 Quick test with {len(test_stocks)} popular stocks...")
        
        quick_results = {}
        for stock in test_stocks:
            result = analyzer.test_single_stock(stock, period)
            if result:
                quick_results[stock] = result
        
        analyzer.results = {'Mixed': quick_results}
        analyzer.analyze_results()
        
    elif scope_choice == '2':
        # Sector analysis
        print("\nAvailable sectors:")
        for i, sector in enumerate(analyzer.stock_universe.keys(), 1):
            print(f"{i}. {sector}")
        
        sector_choice = input("Enter sector number: ").strip()
        try:
            sector_idx = int(sector_choice) - 1
            sector_name = list(analyzer.stock_universe.keys())[sector_idx]
            sector_results = analyzer.test_sector(sector_name, period)
            analyzer.results = {sector_name: sector_results}
            analyzer.analyze_results()
        except (ValueError, IndexError):
            print("Invalid sector choice!")
            
    else:
        # Full analysis
        print(f"\n🔄 Running full analysis on all stocks...")
        print("This may take several minutes...")
        analyzer.test_all_stocks(period)
        analyzer.analyze_results()

if __name__ == "__main__":
    main()