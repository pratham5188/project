#!/usr/bin/env python3
"""
Test script for the Stock Discovery System
This script demonstrates how the automatic stock discovery works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.stock_discovery import StockDiscovery, get_latest_stock_list, auto_update_stocks
from config.settings import INDIAN_STOCKS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stock_discovery():
    """Test the stock discovery functionality"""
    print("=" * 60)
    print("🚀 StockTrendAI - Stock Discovery System Test")
    print("=" * 60)
    
    # Initialize the discovery system
    discovery = StockDiscovery()
    
    print(f"\n📊 Current stocks in config: {len(INDIAN_STOCKS)}")
    
    # Test getting latest stock list
    print("\n🔍 Testing stock discovery...")
    try:
        latest_stocks = get_latest_stock_list()
        print(f"✅ Discovered {len(latest_stocks)} total stocks")
        
        # Find new stocks
        new_stocks = {}
        for symbol, name in latest_stocks.items():
            if symbol not in INDIAN_STOCKS:
                new_stocks[symbol] = name
        
        if new_stocks:
            print(f"\n🆕 Found {len(new_stocks)} new stocks:")
            for symbol, name in list(new_stocks.items())[:10]:  # Show first 10
                print(f"   • {symbol}: {name}")
            if len(new_stocks) > 10:
                print(f"   ... and {len(new_stocks) - 10} more")
        else:
            print("\n✅ No new stocks found - all stocks are up to date")
    
    except Exception as e:
        print(f"❌ Error during discovery: {e}")
    
    # Test auto-update functionality
    print("\n🔄 Testing auto-update system...")
    try:
        success, message, count = auto_update_stocks()
        if success:
            print(f"✅ Auto-update successful: {message}")
            if count > 0:
                print(f"🆕 Added {count} new stocks to configuration")
        else:
            print(f"❌ Auto-update failed: {message}")
    
    except Exception as e:
        print(f"❌ Error during auto-update: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

def test_individual_methods():
    """Test individual discovery methods"""
    print("\n🧪 Testing Individual Discovery Methods")
    print("-" * 40)
    
    discovery = StockDiscovery()
    
    # Test NSE API method
    print("\n1. Testing NSE API method...")
    try:
        nse_stocks = discovery._fetch_nse_api_stocks()
        print(f"   Found {len(nse_stocks)} stocks from NSE API")
    except Exception as e:
        print(f"   ❌ NSE API failed: {e}")
    
    # Test Yahoo Finance method
    print("\n2. Testing Yahoo Finance method...")
    try:
        yahoo_stocks = discovery._fetch_yahoo_finance_stocks()
        print(f"   Found {len(yahoo_stocks)} stocks from Yahoo Finance")
    except Exception as e:
        print(f"   ❌ Yahoo Finance failed: {e}")
    
    # Test web scraping method
    print("\n3. Testing web scraping method...")
    try:
        web_stocks = discovery._fetch_web_scraped_stocks()
        print(f"   Found {len(web_stocks)} stocks from web scraping")
    except Exception as e:
        print(f"   ❌ Web scraping failed: {e}")

def show_sample_stocks():
    """Show sample of current and discovered stocks"""
    print("\n📋 Sample of Current Stocks:")
    print("-" * 30)
    
    sample_stocks = list(INDIAN_STOCKS.items())[:10]
    for symbol, name in sample_stocks:
        print(f"   {symbol}: {name}")
    
    print(f"\n   ... and {len(INDIAN_STOCKS) - 10} more stocks")

if __name__ == "__main__":
    # Show current stocks
    show_sample_stocks()
    
    # Run main test
    test_stock_discovery()
    
    # Test individual methods if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--detailed":
        test_individual_methods()
    
    print("\n💡 Tips:")
    print("   • Run with --detailed flag to test individual methods")
    print("   • The system caches results for 24 hours")
    print("   • New stocks are automatically validated before adding")
    print("   • Configuration file is updated automatically when new stocks are found")