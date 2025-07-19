#!/usr/bin/env python3
"""
Quick Backtest Runner for StockTrendAI Project
==============================================

This script provides a simple way to run backtests on Indian stocks
using the existing project structure and configuration.

Usage:
    python3 run_backtest.py
"""

import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def run_quick_backtest():
    """Run a quick backtest demonstration"""
    
    print("🚀 StockTrendAI - Quick Backtest Runner")
    print("=" * 50)
    
    # Check if we have the necessary files
    if not os.path.exists('backtest_simple.py'):
        print("❌ backtest_simple.py not found!")
        print("Please make sure the backtesting script is in the current directory.")
        return
    
    # Get user input for stock and period
    print("\n📊 Backtest Configuration:")
    stock_symbol = input("Enter stock symbol (default: RELIANCE): ").strip() or "RELIANCE"
    
    period_options = {
        '1': '6mo',
        '2': '1y', 
        '3': '2y',
        '4': '5y'
    }
    
    print("\nSelect time period:")
    print("1. 6 months")
    print("2. 1 year")
    print("3. 2 years")
    print("4. 5 years")
    
    period_choice = input("Enter choice (1-4, default: 2): ").strip() or '2'
    period = period_options.get(period_choice, '1y')
    
    # Ask for detailed analysis
    detailed = input("\nShow detailed analysis? (y/N): ").strip().lower() == 'y'
    
    print(f"\n🔄 Running backtest for {stock_symbol} over {period}...")
    print("-" * 50)
    
    # Build command
    cmd = f"python3 backtest_simple.py --symbol {stock_symbol} --period {period}"
    if detailed:
        cmd += " --detailed"
    
    # Run the backtest
    os.system(cmd)
    
    print("\n" + "=" * 50)
    print("✅ Backtest completed!")
    
    # Ask if user wants to run another test
    another = input("\nRun another backtest? (y/N): ").strip().lower()
    if another == 'y':
        print("\n")
        run_quick_backtest()

def run_multiple_stocks_test():
    """Run backtests on multiple popular Indian stocks"""
    
    print("🚀 StockTrendAI - Multiple Stocks Backtest")
    print("=" * 50)
    
    # Popular Indian stocks
    stocks = [
        'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 
        'HINDUNILVR', 'ITC', 'KOTAKBANK', 'LT', 'ASIANPAINT'
    ]
    
    print("📊 Testing multiple stocks...")
    print(f"Stocks: {', '.join(stocks)}")
    print("Period: 1 year")
    print("-" * 50)
    
    for stock in stocks:
        print(f"\n🔄 Testing {stock}...")
        cmd = f"python3 backtest_simple.py --symbol {stock} --period 1y"
        os.system(cmd)
        print("-" * 30)
    
    print("\n✅ All stocks tested!")

def main():
    """Main menu"""
    
    print("🚀 StockTrendAI Backtesting Menu")
    print("=" * 40)
    print("1. Single stock backtest")
    print("2. Multiple stocks test")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            run_quick_backtest()
            break
        elif choice == '2':
            run_multiple_stocks_test()
            break
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()