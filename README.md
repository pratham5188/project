

# StockTrendAI - Indian Market Predictor

A comprehensive AI-powered stock market prediction application for Indian NSE stocks with futuristic dark theme styling.

## Features

- 🔮 AI-powered stock predictions using XGBoost and LSTM models
- 📊 Real-time market data and technical analysis
- 🟢🔴 Live NSE market status indicator
- 📈 Advanced charting with technical indicators
- 📰 News sentiment analysis
- 💼 Portfolio management
- 🌙 Consistent dark theme across all environments

## Quick Start

1. **Install Python 3.11+** and required packages:
   ```bash
   pip install streamlit yfinance pandas numpy plotly scikit-learn textblob pytz
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Access the app**: Open http://localhost:8501 in your browser

## Environment Consistency

The application is designed to look **exactly the same** whether running on:
- Replit (preview)
- Local machine (localhost)
- Any other hosting environment

### Why it stays consistent:

1. **Forced Dark Theme**: Custom CSS with `!important` rules override any system defaults
2. **Streamlit Config**: `.streamlit/config.toml` enforces dark theme settings
3. **Chart Backgrounds**: All Plotly charts explicitly set dark backgrounds
4. **Font Consistency**: Orbitron font family is loaded and applied globally

### If you see white backgrounds or wrong colors:

1. **Clear browser cache** and reload
2. **Check console errors** for font loading issues
3. **Verify config**: Make sure `.streamlit/config.toml` is in the project root
4. **Restart Streamlit** after any configuration changes

## Project Structure

```
├── app.py                 # Main application
├── config/               
│   └── settings.py       # Application settings
├── models/               
│   ├── xgboost_model.py  # XGBoost prediction model
│   └── lstm_model.py     # LSTM prediction model
├── utils/                
│   ├── data_fetcher.py   # Stock data retrieval
│   ├── technical_indicators.py  # Technical analysis
│   ├── news_sentiment.py # Sentiment analysis
│   └── portfolio_tracker.py    # Portfolio management
├── styles/               
│   └── custom_css.py     # Dark theme styling
├── .streamlit/           
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## Key Features

### Real-time Market Status
- Shows NSE market open/closed status
- Displays current IST time
- Trading hours: 9:15 AM - 3:30 PM IST (Mon-Fri)

### AI Predictions
- XGBoost classification model for direction prediction
- LSTM for time series analysis
- Confidence scoring for predictions
- Technical indicator integration

### Dark Theme
- Futuristic neon green/blue color scheme
- Orbitron monospace font
- Consistent across all devices and browsers
- Environment-independent styling

## Support

If the application doesn't look like the preview:
1. Ensure all files are properly downloaded
2. Run `streamlit run app.py` from the project root
3. Check browser console for any errors
4. Clear cache and reload the page

The styling is designed to be identical everywhere - if you see differences, it's likely a browser cache or file loading issue.
# StockTrendAI


# StockTrendAI

