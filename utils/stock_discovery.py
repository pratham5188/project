"""
Automatic Stock Discovery and Management System
This module handles automatic discovery of new stocks/companies and updates the configuration
"""

import requests
import pandas as pd
import json
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional
import os
import streamlit as st
from config.settings import INDIAN_STOCKS
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDiscovery:
    """Automatic stock discovery and management system"""
    
    def __init__(self):
        self.cache_duration = 86400  # 24 hours cache for stock lists
        self.cache_file = "data/stock_cache.json"
        self.discovered_stocks = {}
        self.last_update = None
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs("data", exist_ok=True)
    
    def get_nse_stock_list(self) -> Dict[str, str]:
        """
        Fetch complete list of NSE stocks using multiple methods
        Returns dictionary of {symbol: company_name}
        """
        all_stocks = {}
        
        try:
            # Method 1: Use NSE India API (if accessible)
            logger.info("Fetching NSE stock list...")
            nse_stocks = self._fetch_nse_api_stocks()
            if nse_stocks:
                all_stocks.update(nse_stocks)
                logger.info(f"Found {len(nse_stocks)} stocks from NSE API")
        except Exception as e:
            logger.warning(f"NSE API method failed: {e}")
        
        try:
            # Method 2: Use predefined Nifty indices to expand our list
            nifty_stocks = self._fetch_nifty_indices_stocks()
            if nifty_stocks:
                all_stocks.update(nifty_stocks)
                logger.info(f"Found {len(nifty_stocks)} stocks from Nifty indices")
        except Exception as e:
            logger.warning(f"Nifty indices method failed: {e}")
        
        try:
            # Method 3: Use Yahoo Finance screening
            yahoo_stocks = self._fetch_yahoo_finance_stocks()
            if yahoo_stocks:
                all_stocks.update(yahoo_stocks)
                logger.info(f"Found {len(yahoo_stocks)} stocks from Yahoo Finance")
        except Exception as e:
            logger.warning(f"Yahoo Finance method failed: {e}")
        
        try:
            # Method 4: Use web scraping for additional stocks
            scraped_stocks = self._fetch_web_scraped_stocks()
            if scraped_stocks:
                all_stocks.update(scraped_stocks)
                logger.info(f"Found {len(scraped_stocks)} stocks from web scraping")
        except Exception as e:
            logger.warning(f"Web scraping method failed: {e}")
        
        # Method 5: Always include our existing stocks
        all_stocks.update(INDIAN_STOCKS)
        logger.info(f"Total unique stocks discovered: {len(all_stocks)}")
        
        return all_stocks
    
    def _fetch_nse_api_stocks(self) -> Dict[str, str]:
        """Fetch stocks from NSE API endpoints"""
        stocks = {}
        
        # NSE endpoints that might contain stock lists
        endpoints = [
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20TOTAL%20MARKET",
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20250",
            "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20150"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        for item in data['data']:
                            symbol = item.get('symbol', '').strip()
                            company = item.get('meta', {}).get('companyName', '') or item.get('companyName', '')
                            
                            if symbol and company:
                                # Clean symbol and add .NS suffix if needed
                                clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                                stocks[clean_symbol] = company.strip()
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {endpoint}: {e}")
                continue
        
        return stocks
    
    def _fetch_nifty_indices_stocks(self) -> Dict[str, str]:
        """Fetch stocks from various Nifty indices"""
        stocks = {}
        
        # Common Nifty indices
        indices = [
            "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
            "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150",
            "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100", "NIFTY SMALLCAP 250",
            "NIFTY TOTAL MARKET", "NIFTY BANK", "NIFTY AUTO", "NIFTY IT",
            "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL", "NIFTY REALTY"
        ]
        
        for index in indices:
            try:
                # Use Yahoo Finance to get index constituents
                index_symbol = self._get_index_symbol(index)
                if index_symbol:
                    ticker = yf.Ticker(index_symbol)
                    info = ticker.info
                    
                    # Try to get holdings or constituents if available
                    if hasattr(ticker, 'recommendations'):
                        recommendations = ticker.recommendations
                        if recommendations is not None:
                            for _, row in recommendations.iterrows():
                                firm = row.get('firm', '')
                                if firm and '.' not in firm:  # Likely a stock symbol
                                    stocks[firm] = f"{firm} (Auto-discovered)"
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Could not fetch constituents for {index}: {e}")
                continue
        
        return stocks
    
    def _fetch_yahoo_finance_stocks(self) -> Dict[str, str]:
        """Fetch Indian stocks using Yahoo Finance screening"""
        stocks = {}
        
        try:
            # Common Indian stock symbols pattern search
            # Search for stocks with .NS suffix (NSE) and .BO suffix (BSE)
            common_patterns = [
                'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
                'ASIANPAINT', 'ITC', 'HINDUNILVR', 'MARUTI', 'KOTAKBANK', 'LT', 'AXISBANK',
                'BAJFINANCE', 'TATAMOTORS', 'SUNPHARMA', 'ULTRACEMCO', 'ONGC', 'NTPC',
                'WIPRO', 'TECHM', 'TATASTEEL', 'POWERGRID', 'NESTLEIND', 'TITAN', 'DRREDDY',
                'CIPLA', 'GRASIM', 'COALINDIA', 'BAJAJFINSV', 'HEROMOTOCO', 'EICHERMOT',
                'DIVISLAB', 'BRITANNIA', 'JSWSTEEL', 'SHREECEM', 'INDUSINDBK', 'ADANIENT',
                'ADANIPORTS', 'HCLTECH', 'HINDALCO', 'VEDL', 'UPL', 'BPCL', 'IOC'
            ]
            
            # Expand search by testing variations
            for base_symbol in common_patterns:
                for suffix in ['.NS', '.BO']:
                    try:
                        full_symbol = base_symbol + suffix
                        ticker = yf.Ticker(full_symbol)
                        info = ticker.info
                        
                        company_name = (info.get('longName') or 
                                      info.get('shortName') or 
                                      base_symbol)
                        
                        if company_name and company_name != base_symbol:
                            stocks[base_symbol] = company_name
                            break  # Found valid company, no need to try other suffix
                        
                    except Exception:
                        continue
                
                time.sleep(0.1)  # Small delay to avoid rate limiting
            
            # Try to discover new stocks by pattern matching
            self._discover_new_symbols(stocks)
            
        except Exception as e:
            logger.warning(f"Yahoo Finance stock discovery failed: {e}")
        
        return stocks
    
    def _fetch_web_scraped_stocks(self) -> Dict[str, str]:
        """Fetch stocks using web scraping (as fallback)"""
        stocks = {}
        
        try:
            # Use BSE/NSE websites for additional discovery
            # This is a simplified implementation - in production, you might want more sophisticated scraping
            
            # BSE stock list endpoint
            bse_url = "https://api.bseindia.com/BseIndiaAPI/api/StockReachGraph/w"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(bse_url, headers=headers, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        for item in data:
                            symbol = item.get('symbol', '').strip()
                            company = item.get('companyName', '').strip()
                            
                            if symbol and company and len(symbol) <= 20:  # Reasonable symbol length
                                clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                                stocks[clean_symbol] = company
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"Web scraping method failed: {e}")
        
        return stocks
    
    def _discover_new_symbols(self, existing_stocks: Dict[str, str]):
        """Discover new stock symbols by pattern matching and validation"""
        
        # Generate potential new symbols based on patterns
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Test some common 3-4 letter combinations that might be valid stocks
        potential_symbols = []
        
        # Generate based on existing patterns
        for existing_symbol in list(existing_stocks.keys())[:10]:  # Test first 10 to avoid too many requests
            # Try variations
            if len(existing_symbol) >= 3:
                base = existing_symbol[:3]
                for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    potential_symbols.append(base + letter)
                    if len(potential_symbols) > 50:  # Limit to avoid excessive requests
                        break
        
        # Test these potential symbols
        for symbol in potential_symbols[:20]:  # Limit to first 20 to avoid rate limiting
            try:
                if symbol not in existing_stocks:
                    # Test if symbol exists
                    ticker = yf.Ticker(f"{symbol}.NS")
                    info = ticker.info
                    
                    company_name = info.get('longName') or info.get('shortName')
                    if company_name and 'symbol' not in company_name.lower():
                        existing_stocks[symbol] = company_name
                        logger.info(f"Discovered new stock: {symbol} - {company_name}")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception:
                continue
    
    def _get_index_symbol(self, index_name: str) -> Optional[str]:
        """Convert index name to Yahoo Finance symbol"""
        index_mapping = {
            "NIFTY 50": "^NSEI",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY METAL": "^CNXMETAL",
            "NIFTY REALTY": "^CNXREALTY"
        }
        return index_mapping.get(index_name)
    
    def validate_new_stocks(self, stocks_dict: Dict[str, str]) -> Dict[str, str]:
        """Validate discovered stocks to ensure they're real and tradeable"""
        validated_stocks = {}
        
        logger.info(f"Validating {len(stocks_dict)} discovered stocks...")
        
        for symbol, company_name in stocks_dict.items():
            try:
                # Test with .NS suffix
                test_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
                ticker = yf.Ticker(test_symbol)
                
                # Try to get recent data
                hist = ticker.history(period="5d")
                info = ticker.info
                
                # Validation criteria
                if (not hist.empty and 
                    len(hist) > 0 and 
                    'longName' in info and 
                    info.get('currency') == 'INR'):
                    
                    # Clean up symbol (remove .NS for our internal use)
                    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                    validated_name = info.get('longName', company_name)
                    
                    validated_stocks[clean_symbol] = validated_name
                    logger.debug(f"Validated: {clean_symbol} - {validated_name}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.debug(f"Validation failed for {symbol}: {e}")
                continue
        
        logger.info(f"Validated {len(validated_stocks)} stocks successfully")
        return validated_stocks
    
    def save_stock_cache(self, stocks: Dict[str, str]):
        """Save discovered stocks to cache"""
        try:
            cache_data = {
                'stocks': stocks,
                'timestamp': datetime.now().isoformat(),
                'total_count': len(stocks)
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(stocks)} stocks to cache")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_stock_cache(self) -> Optional[Dict[str, str]]:
        """Load stocks from cache if recent enough"""
        try:
            if not os.path.exists(self.cache_file):
                return None
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid (24 hours)
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                logger.info(f"Loaded {len(cache_data['stocks'])} stocks from cache")
                return cache_data['stocks']
            else:
                logger.info("Cache expired, will refresh stock list")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def get_updated_stock_list(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get updated stock list, using cache if available or refreshing if needed
        """
        if not force_refresh:
            cached_stocks = self.load_stock_cache()
            if cached_stocks:
                return cached_stocks
        
        logger.info("Refreshing stock list from sources...")
        
        # Discover stocks from all sources
        discovered_stocks = self.get_nse_stock_list()
        
        # Validate the stocks
        validated_stocks = self.validate_new_stocks(discovered_stocks)
        
        # Save to cache
        self.save_stock_cache(validated_stocks)
        
        return validated_stocks
    
    def get_new_stocks_since_last_update(self) -> List[Dict[str, str]]:
        """Get stocks that are new since last update"""
        current_stocks = self.get_updated_stock_list()
        
        new_stocks = []
        for symbol, company_name in current_stocks.items():
            if symbol not in INDIAN_STOCKS:
                new_stocks.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'discovered_date': datetime.now().strftime('%Y-%m-%d')
                })
        
        return new_stocks
    
    def update_config_file(self, new_stocks: Dict[str, str]) -> bool:
        """Update the configuration file with new stocks"""
        try:
            # Read current config
            config_path = "config/settings.py"
            
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the INDIAN_STOCKS dictionary
            start_marker = "INDIAN_STOCKS = {"
            end_marker = "}"
            
            start_idx = content.find(start_marker)
            if start_idx == -1:
                logger.error("Could not find INDIAN_STOCKS in config file")
                return False
            
            # Find the end of the dictionary
            brace_count = 0
            end_idx = start_idx + len(start_marker)
            for i in range(start_idx + len(start_marker), len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    if brace_count == 0:
                        end_idx = i
                        break
                    brace_count -= 1
            
            # Combine existing and new stocks
            all_stocks = dict(INDIAN_STOCKS)
            all_stocks.update(new_stocks)
            
            # Generate new dictionary content
            new_dict_content = "INDIAN_STOCKS = {\n"
            for symbol, company_name in sorted(all_stocks.items()):
                # Escape quotes in company names
                safe_name = company_name.replace("'", "\\'")
                new_dict_content += f"    '{symbol}': '{safe_name}',\n"
            new_dict_content += "}"
            
            # Replace in content
            new_content = content[:start_idx] + new_dict_content + content[end_idx + 1:]
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Updated config file with {len(new_stocks)} new stocks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config file: {e}")
            return False

# Global instance for easy access
stock_discovery = StockDiscovery()

def get_latest_stock_list() -> Dict[str, str]:
    """Get the latest stock list (main function for external use)"""
    return stock_discovery.get_updated_stock_list()

def refresh_stock_list() -> Dict[str, str]:
    """Force refresh of stock list"""
    return stock_discovery.get_updated_stock_list(force_refresh=True)

def get_new_companies() -> List[Dict[str, str]]:
    """Get companies that are new since last check"""
    return stock_discovery.get_new_stocks_since_last_update()

def auto_update_stocks() -> Dict:
    """
    Automatically update stock list and return status
    Returns: {'success': bool, 'message': str, 'new_stocks_count': int}
    """
    try:
        new_stocks = stock_discovery.get_new_stocks_since_last_update()
        
        if not new_stocks:
            return {
                'success': True, 
                'message': "No new stocks found", 
                'new_stocks_count': 0
            }
        
        # Convert to dict format
        new_stocks_dict = {stock['symbol']: stock['company_name'] for stock in new_stocks}
        
        # Update config file
        success = stock_discovery.update_config_file(new_stocks_dict)
        
        if success:
            return {
                'success': True, 
                'message': f"Successfully added {len(new_stocks)} new stocks", 
                'new_stocks_count': len(new_stocks)
            }
        else:
            return {
                'success': False, 
                'message': "Failed to update configuration file", 
                'new_stocks_count': len(new_stocks),
                'error': "Configuration update failed"
            }
            
    except Exception as e:
        return {
            'success': False, 
            'message': f"Auto-update failed: {str(e)}", 
            'new_stocks_count': 0,
            'error': str(e)
        }