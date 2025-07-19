# Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. RELIANCE.NS Error: "possibly delisted; no price data found"

#### **Problem**:
```
ERROR:yfinance:$RELIANCE.NS: possibly delisted; no price data found (period=1d) (Yahoo error = "No data found, symbol may be delisted")
```

#### **Root Cause**:
- **Double .NS suffix**: The symbol might be getting double-appended (e.g., `RELIANCE.NS.NS`)
- **Manual entry**: You might be manually typing "RELIANCE.NS" instead of using the dropdown
- **Caching issues**: Old cached data with incorrect symbol format

#### **Solutions**:

##### ✅ **Use the Dropdown Menu (Recommended)**
1. Go to **Portfolio Tracker** tab
2. In **"Add New Holding"** section
3. **Use the dropdown** that shows "Reliance Industries Ltd (RELIANCE)"
4. **DO NOT** manually type "RELIANCE.NS"
5. The system automatically handles the correct symbol format

##### ✅ **Clear Browser Cache**
1. Refresh the page (Ctrl+F5 or Cmd+Shift+R)
2. Clear browser cache and cookies
3. Restart the Streamlit application

##### ✅ **Check Symbol Format**
- **Correct**: `RELIANCE` (use dropdown)
- **Correct**: `RELIANCE.NS` (system adds this automatically)
- **Incorrect**: `RELIANCE.NS.NS` (double suffix - now fixed)

#### **What We Fixed**:
- ✅ Added symbol cleaning to prevent double `.NS` suffixes
- ✅ Added validation before adding holdings
- ✅ Added helpful error messages and tips
- ✅ Improved symbol handling across all data fetcher methods

---

### 2. Portfolio Holdings Not Updating

#### **Problem**:
- Portfolio shows old prices
- P&L calculations are incorrect
- Holdings don't reflect current market prices

#### **Solutions**:
1. **Refresh the page** to trigger price updates
2. **Check internet connection** - data comes from Yahoo Finance
3. **Wait a few minutes** - prices update automatically
4. **Use the "Refresh Data" button** if available

---

### 3. "No predictions available" Error

#### **Problem**:
- AI prediction cards show "No predictions available"
- Models not generating predictions

#### **Solutions**:
1. **Select at least one AI model** in the sidebar
2. **Check stock symbol** - make sure it's valid
3. **Wait for models to load** - some models take time to initialize
4. **Refresh the page** if models are stuck

---

### 4. Chart Not Loading

#### **Problem**:
- Stock charts don't display
- "Unable to fetch stock data" error

#### **Solutions**:
1. **Check internet connection**
2. **Verify stock symbol** is in our database
3. **Try a different time period** (1d, 1w, 1m, 1y)
4. **Clear browser cache**

---

### 5. Watchlist/Price Alerts Not Working

#### **Problem**:
- Can't add stocks to watchlist
- Price alerts not triggering

#### **Solutions**:
1. **Use dropdown menus** - don't type symbols manually
2. **Check symbol format** - should be like "RELIANCE" not "RELIANCE.NS"
3. **Refresh the page** to see updates
4. **Check if stock is in our database**

---

## 🔧 Technical Details

### Symbol Format Rules:
```
✅ RELIANCE     → RELIANCE.NS (automatic)
✅ RELIANCE.NS  → RELIANCE.NS (kept as is)
❌ RELIANCE.NS.NS → RELIANCE.NS (now fixed)
```

### Valid Stock Symbols:
The application supports **50+ major Indian stocks** including:
- **NIFTY 50 constituents**: RELIANCE, TCS, HDFCBANK, INFY, etc.
- **Banking**: ICICIBANK, KOTAKBANK, SBIN, AXISBANK
- **IT**: HCLTECH, WIPRO, TECHM, LTIM
- **Automotive**: MARUTI, TATAMOTORS, BAJAJ-AUTO
- **Pharmaceuticals**: SUNPHARMA, DRREDDY, CIPLA

### Data Source:
- **Primary**: Yahoo Finance (yfinance library)
- **Exchange**: National Stock Exchange (NSE)
- **Currency**: Indian Rupees (₹)
- **Update Frequency**: Real-time (with 1-5 minute cache)

---

## 📞 Getting Help

### If Issues Persist:
1. **Check the console/logs** for detailed error messages
2. **Try a different stock** to isolate the issue
3. **Restart the application** completely
4. **Check GitHub issues** for known problems

### Common Error Messages:
```
✅ "Added X shares of RELIANCE to portfolio" - Success
❌ "Invalid symbol: RELIANCE.NS" - Use dropdown instead
❌ "No data found" - Check internet connection
❌ "Symbol may be delisted" - Use dropdown menu
```

---

## 🎯 Best Practices

### For Adding Holdings:
1. ✅ **Always use the dropdown menu**
2. ✅ **Don't manually type symbols**
3. ✅ **Let the system handle symbol formatting**
4. ✅ **Check the success message**

### For General Usage:
1. ✅ **Refresh the page if data seems stale**
2. ✅ **Use supported stock symbols only**
3. ✅ **Check internet connection for real-time data**
4. ✅ **Wait for models to load before making predictions**

---

**Last Updated**: December 2024
**Version**: 2.0 (with symbol handling fixes)
**Status**: ✅ All known issues resolved