"""Configuration settings for StockTrendAI application"""

# Indian Stock Symbols (NSE)
INDIAN_STOCKS = {
    'RELIANCE': 'Reliance Industries Ltd',
    'TCS': 'Tata Consultancy Services Ltd',
    'HDFCBANK': 'HDFC Bank Ltd',
    'INFY': 'Infosys Ltd',
    'HINDUNILVR': 'Hindustan Unilever Ltd',
    'ICICIBANK': 'ICICI Bank Ltd',
    'KOTAKBANK': 'Kotak Mahindra Bank Ltd',
    'BHARTIARTL': 'Bharti Airtel Ltd',
    'ITC': 'ITC Ltd',
    'SBIN': 'State Bank of India',
    'BAJFINANCE': 'Bajaj Finance Ltd',
    'HCLTECH': 'HCL Technologies Ltd',
    'ASIANPAINT': 'Asian Paints Ltd',
    'MARUTI': 'Maruti Suzuki India Ltd',
    'LTIM': 'LTIMindtree Ltd',
    'AXISBANK': 'Axis Bank Ltd',
    'WIPRO': 'Wipro Ltd',
    'NESTLEIND': 'Nestle India Ltd',
    'TATAMOTORS': 'Tata Motors Ltd',
    'TECHM': 'Tech Mahindra Ltd',
    'SUNPHARMA': 'Sun Pharmaceutical Industries Ltd',
    'ULTRACEMCO': 'UltraTech Cement Ltd',
    'ONGC': 'Oil & Natural Gas Corporation Ltd',
    'NTPC': 'NTPC Ltd',
    'POWERGRID': 'Power Grid Corporation of India Ltd',
    'COALINDIA': 'Coal India Ltd',
    'TATASTEEL': 'Tata Steel Ltd',
    'TITAN': 'Titan Company Ltd',
    'DRREDDY': 'Dr. Reddys Laboratories Ltd',
    'APOLLOHOSP': 'Apollo Hospitals Enterprise Ltd',
    'ADANIPORTS': 'Adani Ports and Special Economic Zone Ltd',
    'JSWSTEEL': 'JSW Steel Ltd',
    'HINDALCO': 'Hindalco Industries Ltd',
    'INDUSINDBK': 'IndusInd Bank Ltd',
    'BAJAJ-AUTO': 'Bajaj Auto Ltd',
    'CIPLA': 'Cipla Ltd',
    'HEROMOTOCO': 'Hero MotoCorp Ltd',
    'BRITANNIA': 'Britannia Industries Ltd',
    'DIVISLAB': 'Divis Laboratories Ltd',
    'EICHERMOT': 'Eicher Motors Ltd',
    'GRASIM': 'Grasim Industries Ltd',
    'SHRIRAMFIN': 'Shriram Finance Ltd',
    'TRENT': 'Trent Ltd',
    'ADANIENT': 'Adani Enterprises Ltd',
    'BAJAJFINSV': 'Bajaj Finserv Ltd',
    'M&M': 'Mahindra & Mahindra Ltd',
    'BPCL': 'Bharat Petroleum Corporation Ltd',
    'GODREJCP': 'Godrej Consumer Products Ltd',
    'SIEMENS': 'Siemens Ltd',
    'PIDILITIND': 'Pidilite Industries Ltd'
}

# Indian Market Indices (Valid Yahoo Finance symbols)
INDIAN_INDICES = {
    'NIFTYBEES.NS': 'NIFTY 50 ETF',
    'BANKBEES.NS': 'BANK NIFTY ETF',
    'ITBEES.NS': 'IT SECTOR ETF',
    'JUNIORBEES.NS': 'JUNIOR NIFTY ETF',
    'GOLDBEES.NS': 'GOLD ETF',
    'LIQUIDBEES.NS': 'LIQUID ETF',
    'PSUBNKBEES.NS': 'PSU BANK ETF',
    'CPSEETF.NS': 'CPSE ETF'
}

# Default stock to load on startup
DEFAULT_STOCK = 'RELIANCE'

# Model Configuration
MODEL_CONFIG = {
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'lstm': {
        'sequence_length': 60,
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2
    }
}

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    'sma_periods': [20, 50],
    'ema_periods': [20],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'williams_r_period': 14,
    'atr_period': 14,
    'volatility_period': 20,
    'momentum_period': 10,
    'roc_period': 12,
    'cci_period': 20
}

# Data Configuration
DATA_CONFIG = {
    'default_period': '1y',
    'available_periods': ['5m', '15m', '30m', '1h', '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
    'default_interval': '1d',
    'cache_duration': 300,  # 5 minutes in seconds for historical data
    'realtime_cache_duration': 60,  # 1 minute cache for real-time data
    'max_retries': 3,
    'timeout': 30,
    'period_intervals': {
        '5m': '1m',
        '15m': '5m', 
        '30m': '15m',
        '1h': '30m',
        '1d': '1h',
        '5d': '1d',
        '1mo': '1d',
        '3mo': '1d',
        '6mo': '1d',
        '1y': '1d',
        '2y': '1d',
        '5y': '1d',
        '10y': '1d',
        'max': '1d'
    }
}

# UI Configuration
UI_CONFIG = {
    'auto_refresh_interval': 30,  # seconds
    'prediction_card_animation': True,
    'chart_height': 800,
    'neon_glow_enabled': True,
    'mobile_responsive': True
}

# Market Configuration
MARKET_CONFIG = {
    'trading_hours': {
        'start': '09:15',
        'end': '15:30'
    },
    'timezone': 'Asia/Kolkata',
    'market_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'holidays': []  # Can be populated with Indian market holidays
}

# Risk Management
RISK_CONFIG = {
    'max_position_size': 0.1,  # 10% of portfolio
    'stop_loss': 0.05,  # 5% stop loss
    'take_profit': 0.15,  # 15% take profit
    'risk_free_rate': 0.06,  # 6% annual risk-free rate
    'max_drawdown_threshold': 0.2  # 20% maximum drawdown
}

# API Configuration
API_CONFIG = {
    'yahoo_finance': {
        'base_url': 'https://query1.finance.yahoo.com',
        'timeout': 30,
        'max_retries': 3
    },
    'rate_limits': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/stocktrendai.log',
    'max_file_size': 10485760,  # 10MB
    'backup_count': 5
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'minimum_accuracy': 0.55,  # 55% minimum accuracy
    'confidence_threshold': 0.6,  # 60% minimum confidence
    'data_freshness': 3600,  # 1 hour maximum data age
    'model_retrain_threshold': 0.05  # Retrain if accuracy drops by 5%
}

# Feature Engineering
FEATURE_CONFIG = {
    'technical_indicators': True,
    'price_ratios': True,
    'volume_analysis': True,
    'volatility_measures': True,
    'momentum_indicators': True,
    'trend_indicators': True,
    'oscillators': True
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'ensemble_weights': {
        'xgboost': 0.6,
        'lstm': 0.4
    },
    'confidence_adjustment': {
        'volatility_factor': 0.8,
        'trend_factor': 1.2,
        'volume_factor': 1.1
    },
    'prediction_horizon': 1,  # Days
    'update_frequency': 'daily'
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'initial_capital': 100000,  # INR 1 Lakh
    'commission': 0.001,  # 0.1% commission
    'slippage': 0.0005,  # 0.05% slippage
    'min_periods': 252,  # Minimum 1 year of data
    'benchmark': '^NSEI'  # NIFTY 50 as benchmark
}

# Alert Configuration
ALERT_CONFIG = {
    'price_change_threshold': 0.05,  # 5% price change
    'volume_spike_threshold': 2.0,  # 2x average volume
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'volatility_spike_threshold': 1.5  # 1.5x average volatility
}

# Model Persistence
MODEL_PERSISTENCE = {
    'save_interval': 'daily',
    'max_model_age': 30,  # days
    'backup_models': True,
    'compression': True,
    'encryption': False
}
