#!/usr/bin/env python3
"""
Parameter Optimization for StockTrendAI Strategies
==================================================

This script optimizes strategy parameters using grid search and other techniques
to find the best performing parameter combinations.
"""

import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from backtest_simple import MovingAverageStrategy, RSIStrategy, fetch_stock_data
from advanced_strategies import BollingerBandsStrategy, MomentumStrategy

class ParameterOptimizer:
    """Optimize strategy parameters using grid search"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.optimization_results = {}
    
    def optimize_moving_average(self, data: pd.DataFrame, optimization_metric='sharpe_ratio'):
        """Optimize Moving Average strategy parameters"""
        print("🔍 Optimizing Moving Average Strategy Parameters...")
        
        # Parameter ranges to test
        short_windows = [5, 10, 15, 20, 25]
        long_windows = [30, 40, 50, 60, 70]
        
        best_params = None
        best_score = -np.inf
        results = []
        
        total_tests = len(short_windows) * len(long_windows)
        current_test = 0
        
        for short_w in short_windows:
            for long_w in long_windows:
                if short_w >= long_w:  # Skip invalid combinations
                    continue
                
                current_test += 1
                print(f"  Testing {current_test}/{total_tests}: MA({short_w}/{long_w})", end='')
                
                try:
                    strategy = MovingAverageStrategy(short_w, long_w, self.initial_capital)
                    result = strategy.run_backtest(data)
                    metrics = result['metrics']
                    
                    score = metrics.get(optimization_metric, -np.inf)
                    
                    results.append({
                        'short_window': short_w,
                        'long_window': long_w,
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'num_trades': metrics.get('num_trades', 0),
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'short_window': short_w, 'long_window': long_w}
                        print(f" ✅ New best: {score:.3f}")
                    else:
                        print(f" ({score:.3f})")
                
                except Exception as e:
                    print(f" ❌ Error: {e}")
        
        return best_params, best_score, results
    
    def optimize_rsi(self, data: pd.DataFrame, optimization_metric='sharpe_ratio'):
        """Optimize RSI strategy parameters"""
        print("🔍 Optimizing RSI Strategy Parameters...")
        
        # Parameter ranges to test
        rsi_periods = [10, 14, 18, 21]
        oversold_levels = [20, 25, 30, 35]
        overbought_levels = [65, 70, 75, 80]
        
        best_params = None
        best_score = -np.inf
        results = []
        
        total_tests = len(rsi_periods) * len(oversold_levels) * len(overbought_levels)
        current_test = 0
        
        for period in rsi_periods:
            for oversold in oversold_levels:
                for overbought in overbought_levels:
                    if oversold >= overbought:  # Skip invalid combinations
                        continue
                    
                    current_test += 1
                    print(f"  Testing {current_test}/{total_tests}: RSI({period}, {oversold}/{overbought})", end='')
                    
                    try:
                        strategy = RSIStrategy(period, oversold, overbought, self.initial_capital)
                        result = strategy.run_backtest(data)
                        metrics = result['metrics']
                        
                        score = metrics.get(optimization_metric, -np.inf)
                        
                        results.append({
                            'rsi_period': period,
                            'oversold': oversold,
                            'overbought': overbought,
                            'total_return': metrics.get('total_return', 0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                            'max_drawdown': metrics.get('max_drawdown', 0),
                            'win_rate': metrics.get('win_rate', 0),
                            'num_trades': metrics.get('num_trades', 0),
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'rsi_period': period,
                                'oversold': oversold,
                                'overbought': overbought
                            }
                            print(f" ✅ New best: {score:.3f}")
                        else:
                            print(f" ({score:.3f})")
                    
                    except Exception as e:
                        print(f" ❌ Error: {e}")
        
        return best_params, best_score, results
    
    def optimize_bollinger_bands(self, data: pd.DataFrame, optimization_metric='sharpe_ratio'):
        """Optimize Bollinger Bands strategy parameters"""
        print("🔍 Optimizing Bollinger Bands Strategy Parameters...")
        
        # Parameter ranges to test
        windows = [15, 20, 25, 30]
        std_multipliers = [1.5, 2.0, 2.5, 3.0]
        
        best_params = None
        best_score = -np.inf
        results = []
        
        total_tests = len(windows) * len(std_multipliers)
        current_test = 0
        
        for window in windows:
            for std_mult in std_multipliers:
                current_test += 1
                print(f"  Testing {current_test}/{total_tests}: BB({window}, {std_mult})", end='')
                
                try:
                    strategy = BollingerBandsStrategy(window, std_mult, self.initial_capital)
                    result = strategy.run_backtest(data)
                    metrics = result['metrics']
                    
                    score = metrics.get(optimization_metric, -np.inf)
                    
                    results.append({
                        'window': window,
                        'std_multiplier': std_mult,
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'num_trades': metrics.get('num_trades', 0),
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'window': window, 'num_std': std_mult}
                        print(f" ✅ New best: {score:.3f}")
                    else:
                        print(f" ({score:.3f})")
                
                except Exception as e:
                    print(f" ❌ Error: {e}")
        
        return best_params, best_score, results
    
    def optimize_momentum(self, data: pd.DataFrame, optimization_metric='sharpe_ratio'):
        """Optimize Momentum strategy parameters"""
        print("🔍 Optimizing Momentum Strategy Parameters...")
        
        # Parameter ranges to test
        lookbacks = [5, 10, 15, 20, 25]
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1% to 5%
        
        best_params = None
        best_score = -np.inf
        results = []
        
        total_tests = len(lookbacks) * len(thresholds)
        current_test = 0
        
        for lookback in lookbacks:
            for threshold in thresholds:
                current_test += 1
                print(f"  Testing {current_test}/{total_tests}: Momentum({lookback}, {threshold:.2f})", end='')
                
                try:
                    strategy = MomentumStrategy(lookback, threshold, self.initial_capital)
                    result = strategy.run_backtest(data)
                    metrics = result['metrics']
                    
                    score = metrics.get(optimization_metric, -np.inf)
                    
                    results.append({
                        'lookback': lookback,
                        'threshold': threshold,
                        'total_return': metrics.get('total_return', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'num_trades': metrics.get('num_trades', 0),
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'lookback': lookback, 'threshold': threshold}
                        print(f" ✅ New best: {score:.3f}")
                    else:
                        print(f" ({score:.3f})")
                
                except Exception as e:
                    print(f" ❌ Error: {e}")
        
        return best_params, best_score, results
    
    def run_walk_forward_analysis(self, data: pd.DataFrame, strategy_class, best_params: Dict, 
                                optimization_window=252, rebalance_frequency=63):
        """Run walk-forward analysis to test parameter stability"""
        print(f"\n🚶 Running Walk-Forward Analysis...")
        print(f"Optimization window: {optimization_window} days")
        print(f"Rebalance frequency: {rebalance_frequency} days")
        
        if len(data) < optimization_window + rebalance_frequency:
            print("❌ Insufficient data for walk-forward analysis")
            return None
        
        results = []
        start_idx = optimization_window
        
        while start_idx + rebalance_frequency < len(data):
            # Get optimization period
            opt_start = start_idx - optimization_window
            opt_end = start_idx
            opt_data = data.iloc[opt_start:opt_end]
            
            # Get test period
            test_start = start_idx
            test_end = min(start_idx + rebalance_frequency, len(data))
            test_data = data.iloc[test_start:test_end]
            
            print(f"  Period: {data.index[test_start].strftime('%Y-%m-%d')} to {data.index[test_end-1].strftime('%Y-%m-%d')}")
            
            try:
                # Use best parameters (in real implementation, you'd re-optimize here)
                strategy = strategy_class(**best_params, initial_capital=self.initial_capital)
                result = strategy.run_backtest(test_data)
                
                period_result = {
                    'start_date': data.index[test_start],
                    'end_date': data.index[test_end-1],
                    'total_return': result['metrics']['total_return'],
                    'sharpe_ratio': result['metrics']['sharpe_ratio'],
                    'max_drawdown': result['metrics']['max_drawdown'],
                    'num_trades': result['metrics']['num_trades']
                }
                results.append(period_result)
                
                print(f"    Return: {result['metrics']['total_return']:.2f}%, "
                      f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
            
            except Exception as e:
                print(f"    ❌ Error: {e}")
            
            start_idx += rebalance_frequency
        
        # Summarize walk-forward results
        if results:
            avg_return = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            total_trades = sum([r['num_trades'] for r in results])
            
            print(f"\n📊 Walk-Forward Summary:")
            print(f"   Average Return: {avg_return:.2f}%")
            print(f"   Average Sharpe: {avg_sharpe:.3f}")
            print(f"   Total Trades: {total_trades}")
            print(f"   Periods Tested: {len(results)}")
        
        return results
    
    def generate_optimization_report(self, symbol: str, all_results: Dict):
        """Generate comprehensive optimization report"""
        print(f"\n📈 OPTIMIZATION REPORT FOR {symbol}")
        print("=" * 60)
        
        for strategy_name, (best_params, best_score, all_params) in all_results.items():
            print(f"\n🎯 {strategy_name.upper()} STRATEGY:")
            print("-" * 40)
            
            if best_params:
                print(f"✅ Best Parameters: {best_params}")
                print(f"⚡ Best Score: {best_score:.3f}")
                
                # Show parameter sensitivity
                if all_params:
                    df_results = pd.DataFrame(all_params)
                    top_5 = df_results.nlargest(5, 'score')
                    
                    print(f"\n🏆 Top 5 Parameter Combinations:")
                    for i, (_, row) in enumerate(top_5.iterrows(), 1):
                        params_str = ', '.join([f"{k}={v}" for k, v in row.items() 
                                              if k not in ['total_return', 'sharpe_ratio', 'max_drawdown', 
                                                          'win_rate', 'num_trades', 'score']])
                        print(f"   {i}. {params_str} → Score: {row['score']:.3f}, "
                              f"Return: {row['total_return']:.2f}%")
            else:
                print("❌ No valid parameters found")

def main():
    """Main optimization function"""
    print("🚀 StockTrendAI Parameter Optimization")
    print("=" * 50)
    
    # Get user input
    symbol = input("Enter stock symbol (default: RELIANCE): ").strip() or "RELIANCE"
    
    periods = {'1': '1y', '2': '2y', '3': '3y'}
    print("\nSelect data period:")
    print("1. 1 year")
    print("2. 2 years") 
    print("3. 3 years")
    choice = input("Enter choice (1-3, default: 2): ").strip() or '2'
    period = periods.get(choice, '2y')
    
    metrics = {
        '1': 'sharpe_ratio',
        '2': 'total_return', 
        '3': 'win_rate'
    }
    print("\nOptimization metric:")
    print("1. Sharpe Ratio (risk-adjusted)")
    print("2. Total Return")
    print("3. Win Rate")
    metric_choice = input("Enter choice (1-3, default: 1): ").strip() or '1'
    optimization_metric = metrics.get(metric_choice, 'sharpe_ratio')
    
    print(f"\n📊 Fetching data for {symbol}...")
    data = fetch_stock_data(symbol, period)
    if data is None:
        print("❌ Failed to fetch data. Exiting.")
        return
    
    print(f"✅ Data loaded: {len(data)} trading days")
    print(f"🎯 Optimization metric: {optimization_metric}")
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Strategies to optimize
    strategies_to_optimize = {
        'Moving Average': optimizer.optimize_moving_average,
        'RSI': optimizer.optimize_rsi,
        'Bollinger Bands': optimizer.optimize_bollinger_bands,
        'Momentum': optimizer.optimize_momentum
    }
    
    print("\nSelect strategies to optimize:")
    print("1. All strategies")
    print("2. Moving Average only")
    print("3. RSI only")
    print("4. Bollinger Bands only")
    print("5. Momentum only")
    
    strategy_choice = input("Enter choice (1-5, default: 1): ").strip() or '1'
    
    if strategy_choice == '2':
        strategies_to_optimize = {'Moving Average': optimizer.optimize_moving_average}
    elif strategy_choice == '3':
        strategies_to_optimize = {'RSI': optimizer.optimize_rsi}
    elif strategy_choice == '4':
        strategies_to_optimize = {'Bollinger Bands': optimizer.optimize_bollinger_bands}
    elif strategy_choice == '5':
        strategies_to_optimize = {'Momentum': optimizer.optimize_momentum}
    
    # Run optimizations
    print(f"\n🔄 Starting parameter optimization...")
    all_results = {}
    
    for strategy_name, optimize_func in strategies_to_optimize.items():
        print(f"\n" + "="*50)
        try:
            best_params, best_score, all_params = optimize_func(data, optimization_metric)
            all_results[strategy_name] = (best_params, best_score, all_params)
        except Exception as e:
            print(f"❌ Error optimizing {strategy_name}: {e}")
            all_results[strategy_name] = (None, -np.inf, [])
    
    # Generate report
    optimizer.generate_optimization_report(symbol, all_results)
    
    # Optional: Run walk-forward analysis on best strategy
    if all_results:
        best_strategy = max(all_results.items(), key=lambda x: x[1][1])
        strategy_name, (best_params, best_score, _) = best_strategy
        
        if best_params and input(f"\nRun walk-forward analysis on {strategy_name}? (y/N): ").strip().lower() == 'y':
            if strategy_name == 'Moving Average':
                strategy_class = MovingAverageStrategy
            elif strategy_name == 'RSI':
                strategy_class = RSIStrategy
            elif strategy_name == 'Bollinger Bands':
                strategy_class = BollingerBandsStrategy
            elif strategy_name == 'Momentum':
                strategy_class = MomentumStrategy
            
            optimizer.run_walk_forward_analysis(data, strategy_class, best_params)
    
    print(f"\n✅ Parameter optimization completed!")

if __name__ == "__main__":
    main()