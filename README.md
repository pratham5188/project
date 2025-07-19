

# StockTrendAI - Indian Market Predictor

A comprehensive AI-powered stock prediction application for the Indian market, built with Streamlit and featuring multiple machine learning models.

## Features

- **Multi-Model Predictions**: XGBoost, LSTM, Prophet, Transformer, GRU, and Ensemble models
- **Real-time Data**: Live stock data from Yahoo Finance
- **Technical Analysis**: Comprehensive technical indicators
- **Sentiment Analysis**: News sentiment analysis for stocks
- **Portfolio Tracking**: Track your investments and performance
- **Advanced Analytics**: Risk analysis, Monte Carlo simulations, and correlation analysis
- **Beautiful UI**: Futuristic 3D neon glow interface

## Deployment Status

✅ **All deployment issues have been resolved!**

### Fixed Issues:
- ✅ Python version compatibility (updated to Python 3.13)
- ✅ Missing dependencies (JAX, Prophet, Keras, etc.)
- ✅ Import errors resolved
- ✅ Streamlit configuration optimized
- ✅ Deployment scripts created

## Quick Start

### Local Development
```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the application
python3 -m streamlit run app.py
```

### Deployment
```bash
# Use the provided startup script
./start.sh
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── .replit              # Replit deployment config
├── start.sh             # Startup script
├── test_imports.py      # Import verification script
├── utils/               # Utility modules
│   ├── data_fetcher.py
│   ├── technical_indicators.py
│   ├── model_utils.py
│   ├── portfolio_tracker.py
│   ├── advanced_analytics.py
│   ├── news_sentiment.py
│   ├── ui_components.py
│   └── model_info.py
├── models/              # ML model implementations
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── prophet_model.py
│   ├── ensemble_model.py
│   ├── transformer_model.py
│   ├── gru_model.py
│   └── stacking_ensemble.py
├── config/              # Configuration files
│   └── settings.py
└── styles/              # UI styling
    └── custom_css.py
```

## Dependencies

### Core Dependencies
- **Streamlit** >= 1.47.0 - Web application framework
- **Pandas** >= 2.3.1 - Data manipulation
- **NumPy** < 2.0 - Numerical computing
- **Plotly** >= 6.2.0 - Interactive charts

### Machine Learning
- **Scikit-learn** >= 1.7.0 - Traditional ML algorithms
- **XGBoost** >= 3.0.2 - Gradient boosting
- **LightGBM** >= 4.3.0 - Light gradient boosting
- **TensorFlow** >= 2.18.0 - Deep learning framework
- **Keras** >= 3.10.0 - High-level neural networks
- **JAX** >= 0.5.0 - High-performance ML
- **Prophet** >= 1.1.7 - Time series forecasting

### Data Sources
- **YFinance** >= 0.2.65 - Stock market data
- **TextBlob** >= 0.19.0 - Sentiment analysis

## Configuration

### Environment Variables
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export STREAMLIT_SERVER_PORT=5000
export STREAMLIT_SERVER_HEADLESS=true
```

### Streamlit Configuration
The application uses a custom Streamlit configuration in `.streamlit/config.toml`:
- Port: 5000
- Headless mode enabled
- Custom theme with neon colors
- Error details enabled for debugging

## Usage

1. **Select a Stock**: Choose from the dropdown or enter a custom symbol
2. **Choose Time Period**: Select from 1 day to 10 years
3. **Enable Models**: Select which ML models to use for predictions
4. **View Results**: See predictions, charts, and analysis

## Model Information

### XGBoost Model
- **Type**: Gradient Boosting (Random Forest fallback)
- **Features**: Technical indicators, price ratios, volume analysis
- **Use Case**: Short-term price direction prediction

### LSTM Model
- **Type**: Long Short-Term Memory neural network
- **Features**: Sequential price data
- **Use Case**: Pattern recognition in time series

### Prophet Model
- **Type**: Facebook's time series forecasting
- **Features**: Trend and seasonality analysis
- **Use Case**: Long-term trend prediction

### Ensemble Model
- **Type**: Combination of multiple models
- **Features**: Weighted average of predictions
- **Use Case**: Improved accuracy and reliability

## Troubleshooting

### Common Issues

1. **Import Errors**: Run `python3 test_imports.py` to verify all dependencies
2. **Port Conflicts**: Change port in `.streamlit/config.toml`
3. **Memory Issues**: Reduce model complexity in settings
4. **Data Fetching**: Check internet connection and Yahoo Finance availability

### Verification Commands
```bash
# Test all imports
python3 test_imports.py

# Check Python version
python3 --version

# Verify Streamlit installation
python3 -m streamlit --version
```

## Performance Notes

- **GPU Support**: Application automatically detects and uses GPU if available
- **Caching**: Data is cached for 5 minutes to improve performance
- **Model Loading**: Pre-trained models are loaded on startup
- **Memory Usage**: Optimized for deployment environments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test script
3. Review the logs
4. Create an issue with detailed information

---

**Status**: ✅ Ready for deployment
**Last Updated**: July 19, 2025
**Version**: 1.0.0

