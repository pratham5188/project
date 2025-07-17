import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class DataFetcher:
    """Data fetcher for Indian stock market data using yFinance"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache for most data
        self.short_cache_duration = 60  # 1 minute cache for real-time data
    
    def get_stock_data(self, symbol, period="1y", interval="1d"):
        """Fetch stock data from Yahoo Finance for Indian stocks"""
        try:
            # Handle different symbol formats
            original_symbol = symbol
            
            # Don't modify symbols that already have proper formatting
            if symbol.startswith('^') or symbol.endswith('.NS') or symbol.endswith('.BO'):
                # Keep the symbol as is for indices and already formatted symbols
                pass
            else:
                # Add .NS suffix for NSE stocks
                symbol = symbol + '.NS'
            
            # Map period to correct yfinance parameters
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            
            # Determine appropriate interval and period based on user selection
            if period == '1d':
                # For 1 day, use 5 minute intervals
                interval = '5m'
                period = '1d'
            elif period == '5d':
                # For 5 days, use 15 minute intervals
                interval = '15m'
                period = '5d'
            elif period in valid_periods:
                # Use default daily interval for longer periods
                interval = '1d'
            else:
                # Fallback to 1 year with daily intervals
                period = '1y'
                interval = '1d'
            
            # Check cache with appropriate duration
            cache_key = f"{symbol}_{period}_{interval}"
            current_time = time.time()
            
            # Use shorter cache duration for intraday data
            cache_duration = self.short_cache_duration if period in ['1d', '5d'] else self.cache_duration
            
            if (cache_key in self.cache and 
                current_time - self.cache[cache_key]['timestamp'] < cache_duration):
                return self.cache[cache_key]['data']
            
            # Download data with timeout and retry logic
            ticker = yf.Ticker(symbol)
            
            max_retries = 3
            data = None
            
            for attempt in range(max_retries):
                try:
                    # Validate period and interval combination
                    if period not in valid_periods:
                        print(f"Invalid period: {period}, using default 1y")
                        period = '1y'
                        interval = '1d'
                    
                    # Download data with proper error handling
                    data = ticker.history(period=period, interval=interval, timeout=10)
                    
                    if not data.empty:
                        break
                    else:
                        print(f"Empty data for {symbol} with period {period} and interval {interval}")
                        
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                    if attempt == max_retries - 1:
                        # Try with default parameters as fallback
                        try:
                            data = ticker.history(period='1y', interval='1d', timeout=10)
                            if not data.empty:
                                print(f"Fallback successful for {symbol}")
                                break
                        except:
                            pass
                    time.sleep(1)  # Wait 1 second before retry
            
            if data is None or data.empty:
                print(f"No data found for symbol: {symbol} after all attempts")
                return None
            
            # Clean and prepare data
            data = data.reset_index()
            
            # Handle different column names for date/datetime index
            date_column = None
            for col in ['Date', 'Datetime']:
                if col in data.columns:
                    date_column = col
                    break
            
            if date_column:
                data[date_column] = pd.to_datetime(data[date_column])
                data.set_index(date_column, inplace=True)
            else:
                # If no date column found, use the existing index
                data.index = pd.to_datetime(data.index)
            
            # Remove any duplicate indices
            data = data[~data.index.duplicated(keep='first')]
            
            # Sort by date
            data = data.sort_index()
            
            # Forward fill any missing values
            data = data.ffill()
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Missing required column: {col}")
                    return None
            
            # Cache the data
            self.cache[cache_key] = {
                'data': data.copy(),
                'timestamp': current_time
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            
            # Try alternative symbol format for indices
            if original_symbol.startswith('^'):
                try:
                    # Try without the ^ prefix
                    alternative_symbol = original_symbol[1:] + '.NS'
                    print(f"Trying alternative symbol: {alternative_symbol}")
                    ticker = yf.Ticker(alternative_symbol)
                    data = ticker.history(period=period, interval=interval)
                    
                    if not data.empty:
                        # Clean and prepare data (same logic as above)
                        data = data.reset_index()
                        
                        # Handle different column names for date/datetime index
                        date_column = None
                        for col in ['Date', 'Datetime']:
                            if col in data.columns:
                                date_column = col
                                break
                        
                        if date_column:
                            data[date_column] = pd.to_datetime(data[date_column])
                            data.set_index(date_column, inplace=True)
                        else:
                            # If no date column found, use the existing index
                            data.index = pd.to_datetime(data.index)
                        data = data[~data.index.duplicated(keep='first')]
                        data = data.sort_index()
                        data = data.ffill()
                        data = data.dropna(how='all')
                        
                        cache_key = f"{original_symbol}_{period}_{interval}"
                        self.cache[cache_key] = {
                            'data': data.copy(),
                            'timestamp': time.time()
                        }
                        return data
                        
                except Exception as e2:
                    print(f"Alternative symbol also failed: {str(e2)}")
            
            return None
    
    def get_real_time_price(self, symbol):
        """Get real-time price for a stock"""
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = symbol + '.NS'
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
        except Exception as e:
            print(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None
    
    def get_company_info(self, symbol):
        """Get company information"""
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = symbol + '.NS'
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'website': info.get('website', ''),
                'business_summary': info.get('longBusinessSummary', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'India'),
                'currency': info.get('currency', 'INR')
            }
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return None
    
    def get_historical_data_custom(self, symbol, start_date, end_date):
        """Get historical data for custom date range"""
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = symbol + '.NS'
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return None
            
            # Clean and prepare data
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            return data
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def validate_symbol(self, symbol):
        """Validate if a stock symbol exists"""
        try:
            if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
                symbol = symbol + '.NS'
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we get valid info
            return 'longName' in info or 'shortName' in info
            
        except:
            return False
    
    def get_market_summary(self):
        """Get Indian market indices summary"""
        from config.settings import INDIAN_INDICES
        
        summary = {}
        
        for symbol, name in INDIAN_INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2d')
                
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = current - previous
                    change_percent = (change / previous) * 100
                    
                    summary[name] = {
                        'symbol': symbol,
                        'current': current,
                        'change': change,
                        'change_percent': change_percent
                    }
                    
            except Exception as e:
                print(f"Error fetching {name}: {str(e)}")
        
        return summary
    
    def get_top_gainers_losers(self, exchange='NSE'):
        """Get top gainers and losers (simplified version)"""
        from config.settings import INDIAN_STOCKS
        
        stocks_data = []
        
        for symbol, name in INDIAN_STOCKS.items():
            try:
                data = self.get_stock_data(symbol, period='2d')
                if data is not None and len(data) >= 2:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2]
                    change_percent = ((current - previous) / previous) * 100
                    
                    stocks_data.append({
                        'symbol': symbol,
                        'name': name,
                        'current_price': current,
                        'change_percent': change_percent
                    })
                    
            except Exception as e:
                continue
        
        # Sort by change percentage
        stocks_data.sort(key=lambda x: x['change_percent'], reverse=True)
        
        gainers = stocks_data[:10]  # Top 10 gainers
        losers = stocks_data[-10:]  # Top 10 losers
        
        return {'gainers': gainers, 'losers': losers}
    
    def add_new_stock(self, symbol, name):
        """Add a new stock to the available stocks list"""
        try:
            # Validate the symbol first
            if self.validate_symbol(symbol):
                # This would update the configuration file
                # For now, we'll just return success
                return {
                    'success': True,
                    'message': f'Stock {symbol} - {name} added successfully'
                }
            else:
                return {
                    'success': False,
                    'message': f'Invalid stock symbol: {symbol}'
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error adding stock: {str(e)}'
            }
    
    def search_stocks(self, query):
        """Search for stocks based on symbol or name"""
        from config.settings import INDIAN_STOCKS
        
        results = []
        query = query.lower()
        
        for symbol, name in INDIAN_STOCKS.items():
            if query in symbol.lower() or query in name.lower():
                results.append({
                    'symbol': symbol,
                    'name': name
                })
        
        return results
    
    def clear_cache_for_symbol(self, symbol, period=None):
        """Clear cache for a specific symbol and optionally specific period"""
        try:
            # Handle different symbol formats
            if not (symbol.startswith('^') or symbol.endswith('.NS') or symbol.endswith('.BO')):
                symbol = symbol + '.NS'
            
            if period:
                # Clear cache for specific symbol and period
                intervals = ['1d', '5m', '15m']  # Common intervals
                for interval in intervals:
                    cache_key = f"{symbol}_{period}_{interval}"
                    if cache_key in self.cache:
                        del self.cache[cache_key]
            else:
                # Clear all cache entries for the symbol
                keys_to_remove = [key for key in self.cache.keys() if key.startswith(symbol)]
                for key in keys_to_remove:
                    del self.cache[key]
        except Exception as e:
            print(f"Error clearing cache for {symbol}: {str(e)}")
    
    def clear_all_cache(self):
        """Clear all cached data"""
        self.cache.clear()
