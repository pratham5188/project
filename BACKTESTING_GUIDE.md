# StockTrendAI Backtesting Guide 📈

## Overview

This guide explains how to use the comprehensive backtesting functionality added to your StockTrendAI project. The backtesting system allows you to test various trading strategies on historical Indian stock market data and evaluate their performance using professional risk-adjusted metrics.

## 🚀 Quick Start

### Basic Usage

```bash
# Simple backtest for Reliance stock
python3 backtest_simple.py --symbol RELIANCE --period 1y

# Interactive runner with menu
python3 run_backtest.py
```

### Advanced Usage

```bash
# Detailed analysis with specific parameters
python3 backtest_simple.py --symbol TCS --period 2y --capital 150000 --detailed

# Test multiple strategies with full visualization
python3 backtest_comprehensive.py --symbol INFY --save-html
```

## 📁 Files Overview

| File | Description |
|------|-------------|
| `backtest_simple.py` | Main backtesting script with multiple strategies |
| `backtest_comprehensive.py` | Advanced backtesting with visualizations |
| `run_backtest.py` | Interactive runner with user-friendly menu |
| `BACKTESTING_GUIDE.md` | This documentation file |

## 🎯 Available Strategies

### 1. Buy & Hold Strategy
- **Description**: Simple buy at start, sell at end strategy
- **Use Case**: Baseline comparison for other strategies
- **Parameters**: None

### 2. Moving Average Crossover
- **Description**: Buy when short MA crosses above long MA, sell when it crosses below
- **Variants**: 
  - MA (20/50): 20-day vs 50-day moving averages
  - MA (10/30): 10-day vs 30-day moving averages
- **Use Case**: Trend-following strategy

### 3. RSI Mean Reversion
- **Description**: Buy when RSI indicates oversold, sell when overbought
- **Parameters**:
  - RSI Period: 14 days (default)
  - Oversold: 30 (default)
  - Overbought: 70 (default)
- **Variants**:
  - Standard: 30/70 levels
  - Aggressive: 25/75 levels

### 4. MACD Strategy
- **Description**: Buy/sell based on MACD line crossing signal line
- **Parameters**: Fast(12), Slow(26), Signal(9)
- **Use Case**: Momentum trading

## 📊 Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return vs maximum drawdown

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: 95% confidence interval for losses
- **Volatility**: Annualized price volatility

### Trading Metrics
- **Total Return**: Overall percentage return
- **Win Rate**: Percentage of profitable trades
- **Number of Trades**: Total trading frequency

## 🛠️ Usage Examples

### Example 1: Test Single Stock

```bash
python3 backtest_simple.py --symbol HDFCBANK --period 2y --detailed
```

**Output includes:**
- Strategy comparison table
- Detailed metrics for each strategy
- Best strategy identification
- Benchmark comparison

### Example 2: Interactive Testing

```bash
python3 run_backtest.py
```

**Features:**
- User-friendly menu interface
- Stock symbol input with validation
- Time period selection
- Option for detailed analysis

### Example 3: Batch Testing

```python
# Test multiple stocks programmatically
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']
for stock in stocks:
    os.system(f"python3 backtest_simple.py --symbol {stock} --period 1y")
```

## 📈 Sample Output

```
🚀 StockTrendAI Backtesting Engine
============================================================
Symbol: TCS
Period: 2y
Initial Capital: ₹100,000.00
============================================================

📊 BACKTEST RESULTS SUMMARY
====================================================================================================
Strategy                  Return %     Sharpe     Max DD %     Volatility %    Trades     Win Rate %
----------------------------------------------------------------------------------------------------
Buy & Hold                   -5.69%    -0.344     -28.40%         20.15%         2          0.0%
MA Crossover (20/50)         -0.13%    -0.365     -16.03%         13.98%        10         40.0%
MA Crossover (10/30)         21.98%     0.354     -10.45%         14.72%        12         50.0%
RSI Mean Reversion          -12.18%    -0.876     -24.46%         13.39%        11         60.0%
RSI Aggressive              -12.25%    -1.069     -23.62%         11.26%         7         66.7%

🏆 BEST STRATEGIES:
📊 Highest Sharpe Ratio: MA Crossover (10/30) (0.354)
💰 Highest Return:       MA Crossover (10/30) (21.98%)
```

## ⚙️ Configuration Options

### Command Line Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--symbol` | Stock symbol | RELIANCE | `--symbol TCS` |
| `--period` | Time period | 2y | `--period 1y` |
| `--capital` | Initial capital | 100000 | `--capital 150000` |
| `--detailed` | Show detailed analysis | False | `--detailed` |
| `--save-html` | Save charts as HTML | False | `--save-html` |

### Supported Time Periods
- `6mo`: 6 months
- `1y`: 1 year
- `2y`: 2 years (default)
- `5y`: 5 years
- `max`: Maximum available data

### Supported Stock Symbols

All Indian stocks listed on NSE/BSE. The system automatically adds `.NS` suffix for Indian stocks.

**Popular stocks include:**
- RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK
- HINDUNILVR, ITC, KOTAKBANK, LT, ASIANPAINT
- BAJFINANCE, SBIN, WIPRO, MARUTI, NESTLEIND

## 🔧 Technical Details

### Transaction Costs
- **Commission**: 0.1% per trade (configurable)
- **Slippage**: Included in commission for simplicity
- **Market Impact**: Not modeled (suitable for retail trading)

### Data Source
- **Provider**: Yahoo Finance
- **Frequency**: Daily OHLCV data
- **Currency**: Indian Rupees (INR)
- **Market**: NSE (National Stock Exchange)

### Risk-Free Rate
- **Default**: 6% annual (Indian government bond rate)
- **Usage**: Sharpe ratio and Sortino ratio calculations

## 🚨 Important Notes

### Limitations
1. **Historical Performance**: Past performance doesn't guarantee future results
2. **Market Conditions**: Strategies may perform differently in various market conditions
3. **Transaction Costs**: Real trading costs may be higher than modeled
4. **Slippage**: Actual execution prices may differ from backtested prices

### Best Practices
1. **Multiple Timeframes**: Test strategies across different time periods
2. **Out-of-Sample Testing**: Reserve recent data for validation
3. **Risk Management**: Consider position sizing and stop-losses
4. **Market Regime Analysis**: Evaluate performance in bull/bear markets

## 🤝 Integration with StockTrendAI

The backtesting system is designed to complement your existing StockTrendAI project:

### Using with Existing Models
```python
# Example: Integrate with your prediction models
from utils.model_utils import ModelUtils
from backtest_simple import MovingAverageStrategy

# Get predictions from your models
predictions = your_model.predict(stock_data)

# Convert predictions to trading signals
# Then use with backtesting engine
```

### Extending Strategies
```python
class CustomStrategy(BacktestStrategy):
    def generate_signals(self, data):
        # Implement your custom strategy logic
        # Return pd.Series with 1 (buy), -1 (sell), 0 (hold)
        pass
```

## 📞 Support and Troubleshooting

### Common Issues

1. **"No data found"**: Check stock symbol spelling and availability
2. **Import errors**: Ensure all dependencies are installed
3. **Memory issues**: Use shorter time periods for large datasets

### Getting Help

If you encounter issues:
1. Check the error message and this guide
2. Verify stock symbol is valid for Indian markets
3. Ensure internet connection for data fetching
4. Try with a shorter time period

### Sample Debug Commands

```bash
# Test with known working symbol
python3 backtest_simple.py --symbol RELIANCE --period 6mo

# Check data availability
python3 -c "import yfinance as yf; print(yf.Ticker('RELIANCE.NS').history(period='1y').head())"
```

## 🔄 Regular Updates

The backtesting system can be extended with:
- Additional technical indicators
- More sophisticated strategies
- Portfolio optimization features
- Machine learning integration
- Real-time paper trading

---

**Happy Backtesting! 📈🚀**

*Remember: Use this tool to validate strategies before real trading. Always consider your risk tolerance and investment goals.*