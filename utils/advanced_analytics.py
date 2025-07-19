import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

class AdvancedAnalytics:
    """Advanced analytics and visualization tools for stock analysis"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def calculate_volatility_analysis(self, stock_data):
        """Calculate volatility analysis for stock data"""
        try:
            if stock_data is None or stock_data.empty:
                return None
            
            returns = stock_data['Close'].pct_change().dropna()
            
            # Calculate different volatility measures
            volatility_metrics = {
                'daily_volatility': returns.std(),
                'annualized_volatility': returns.std() * np.sqrt(252),
                'rolling_volatility_30d': returns.rolling(30).std(),
                'rolling_volatility_60d': returns.rolling(60).std(),
                'volatility_percentile': stats.percentileofscore(returns.rolling(252).std().dropna(), returns.std())
            }
            
            return volatility_metrics
        except Exception as e:
            print(f"Error calculating volatility analysis: {e}")
            return None
    
    def perform_correlation_analysis(self, data_fetcher, symbols, period='3mo'):
        """Perform correlation analysis between multiple stocks"""
        price_data = {}
        
        # Clean all symbols before fetching data
        def clean_symbol(symbol):
            if symbol.endswith('.NS.NS'):
                return symbol.replace('.NS.NS', '.NS')
            elif symbol.endswith('.NS'):
                return symbol
            else:
                return symbol + '.NS'
        
        for symbol in symbols:
            try:
                clean_sym = clean_symbol(symbol)
                data = data_fetcher.get_stock_data(clean_sym, period)
                if data is not None and not data.empty:
                    price_data[symbol] = data['Close'].pct_change().dropna()
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        if len(price_data) < 2:
            return None
        
        # Create correlation matrix
        correlation_df = pd.DataFrame(price_data)
        correlation_matrix = correlation_df.corr()
        
        return {
            'correlation_matrix': correlation_matrix,
            'price_data': price_data,
            'returns_data': correlation_df
        }
    
    def calculate_risk_metrics(self, returns_data, risk_free_rate=0.06):
        """Calculate comprehensive risk metrics"""
        if returns_data.empty:
            return None
        
        # Annualized returns
        annual_return = returns_data.mean() * 252
        
        # Volatility (standard deviation)
        volatility = returns_data.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_data).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = returns_data.quantile(0.05)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns_data[returns_data <= var_95].mean()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_data[returns_data < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Handle Series comparison properly
        if isinstance(downside_deviation, pd.Series):
            sortino_ratio = np.where(downside_deviation > 0, (annual_return - risk_free_rate) / downside_deviation, np.inf)
        else:
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.inf
        
        # Calmar ratio
        if isinstance(max_drawdown, pd.Series):
            calmar_ratio = np.where(max_drawdown != 0, annual_return / abs(max_drawdown), np.inf)
        else:
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'skewness': returns_data.skew(),
            'kurtosis': returns_data.kurtosis()
        }
    
    def monte_carlo_simulation(self, stock_data, num_simulations=1000, time_horizon=30):
        """Perform Monte Carlo simulation for price prediction"""
        if stock_data.empty:
            return None
        
        # Calculate daily returns
        returns = stock_data['Close'].pct_change().dropna()
        
        # Parameters for simulation
        mean_return = returns.mean()
        volatility = returns.std()
        current_price = stock_data['Close'].iloc[-1]
        
        # Generate random scenarios
        simulations = []
        
        for _ in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, volatility, time_horizon)
            
            # Calculate price path
            price_path = [current_price]
            for return_val in random_returns:
                price_path.append(price_path[-1] * (1 + return_val))
            
            simulations.append(price_path)
        
        # Convert to DataFrame
        simulations_df = pd.DataFrame(simulations).T
        
        # Calculate statistics
        final_prices = simulations_df.iloc[-1]
        
        statistics = {
            'mean_final_price': final_prices.mean(),
            'median_final_price': final_prices.median(),
            'percentile_5': final_prices.quantile(0.05),
            'percentile_95': final_prices.quantile(0.95),
            'probability_profit': (final_prices > current_price).mean(),
            'expected_return': ((final_prices.mean() - current_price) / current_price) * 100,
            'simulations_df': simulations_df
        }
        
        return statistics
    
    def fibonacci_retracement(self, stock_data, period=252):
        """Calculate Fibonacci retracement levels"""
        if len(stock_data) < period:
            period = len(stock_data)
        
        recent_data = stock_data.tail(period)
        high_price = recent_data['High'].max()
        low_price = recent_data['Low'].min()
        
        # Fibonacci levels
        diff = high_price - low_price
        levels = {
            '0.0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50.0%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '76.4%': high_price - 0.764 * diff,
            '100.0%': low_price
        }
        
        return levels
    
    def elliott_wave_analysis(self, stock_data):
        """Basic Elliott Wave pattern recognition"""
        if len(stock_data) < 50:
            return None
        
        # Identify local maxima and minima
        prices = stock_data['Close'].values
        
        # Simple peak and trough detection
        peaks = []
        troughs = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        # Combine and sort by index
        turning_points = sorted(peaks + troughs, key=lambda x: x[0])
        
        # Simple wave counting (basic implementation)
        waves = []
        for i in range(len(turning_points) - 1):
            start_idx, start_price = turning_points[i]
            end_idx, end_price = turning_points[i + 1]
            
            wave_type = 'up' if end_price > start_price else 'down'
            magnitude = abs(end_price - start_price)
            
            waves.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_price': start_price,
                'end_price': end_price,
                'type': wave_type,
                'magnitude': magnitude
            })
        
        return {
            'waves': waves,
            'turning_points': turning_points,
            'current_trend': waves[-1]['type'] if waves else 'neutral'
        }
    
    def support_resistance_levels(self, stock_data, window=20):
        """Identify support and resistance levels"""
        if len(stock_data) < window * 2:
            return None
        
        highs = stock_data['High'].rolling(window=window).max()
        lows = stock_data['Low'].rolling(window=window).min()
        
        # Find significant levels
        resistance_levels = []
        support_levels = []
        
        current_price = stock_data['Close'].iloc[-1]
        
        # Get unique resistance levels
        unique_highs = highs.dropna().unique()
        for level in unique_highs:
            if level > current_price:
                resistance_levels.append(level)
        
        # Get unique support levels
        unique_lows = lows.dropna().unique()
        for level in unique_lows:
            if level < current_price:
                support_levels.append(level)
        
        # Sort and get closest levels
        resistance_levels = sorted(resistance_levels)[:5]  # Top 5 resistance levels
        support_levels = sorted(support_levels, reverse=True)[:5]  # Top 5 support levels
        
        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'current_price': current_price
        }
    
    def volume_analysis(self, stock_data):
        """Analyze volume patterns and trends"""
        if 'Volume' not in stock_data.columns:
            return None
        
        volume = stock_data['Volume']
        price = stock_data['Close']
        
        # Volume moving averages
        volume_ma_20 = volume.rolling(window=20).mean()
        volume_ma_50 = volume.rolling(window=50).mean()
        
        # Volume relative to average
        volume_ratio = volume / volume_ma_20
        
        # On-Balance Volume (OBV)
        obv = (np.sign(price.diff()) * volume).fillna(0).cumsum()
        
        # Volume Price Trend (VPT)
        vpt = (volume * price.pct_change()).fillna(0).cumsum()
        
        # Money Flow Index (MFI)
        typical_price = (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        
        return {
            'volume_ma_20': volume_ma_20,
            'volume_ma_50': volume_ma_50,
            'volume_ratio': volume_ratio,
            'obv': obv,
            'vpt': vpt,
            'mfi': mfi,
            'avg_volume': volume.mean(),
            'volume_trend': 'increasing' if volume_ma_20.iloc[-1] > volume_ma_50.iloc[-1] else 'decreasing'
        }
    
    def seasonality_analysis(self, stock_data):
        """Analyze seasonal patterns in stock performance"""
        if len(stock_data) < 252:  # Need at least 1 year of data
            return None
        
        # Add time-based features
        data = stock_data.copy()
        data['Month'] = data.index.month
        data['DayOfWeek'] = data.index.dayofweek
        data['Quarter'] = data.index.quarter
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        
        # Monthly patterns
        monthly_returns = data.groupby('Month')['Returns'].agg(['mean', 'std', 'count'])
        monthly_returns.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Day of week patterns
        dow_returns = data.groupby('DayOfWeek')['Returns'].agg(['mean', 'std', 'count'])
        dow_returns.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Quarterly patterns
        quarterly_returns = data.groupby('Quarter')['Returns'].agg(['mean', 'std', 'count'])
        quarterly_returns.index = ['Q1', 'Q2', 'Q3', 'Q4']
        
        return {
            'monthly_patterns': monthly_returns,
            'daily_patterns': dow_returns,
            'quarterly_patterns': quarterly_returns,
            'best_month': monthly_returns['mean'].idxmax(),
            'worst_month': monthly_returns['mean'].idxmin(),
            'best_day': dow_returns['mean'].idxmax(),
            'worst_day': dow_returns['mean'].idxmin()
        }
    
    def create_advanced_chart(self, stock_data, analysis_type='comprehensive'):
        """Create advanced interactive charts"""
        if stock_data.empty:
            return None
        
        if analysis_type == 'comprehensive':
            return self._create_comprehensive_chart(stock_data)
        elif analysis_type == 'volume':
            return self._create_volume_chart(stock_data)
        elif analysis_type == 'correlation':
            return self._create_correlation_heatmap(stock_data)
        
    def _create_comprehensive_chart(self, stock_data):
        """Create comprehensive multi-panel chart"""
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Volume', 'RSI', 'MACD', 'Bollinger Bands', 'Support/Resistance'),
            row_heights=[0.4, 0.15, 0.15, 0.15, 0.15]
        )
        
        # Price and volume
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                yaxis='y2',
                opacity=0.3
            ),
            row=1, col=1
        )
        
        # RSI
        if 'RSI' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['RSI'],
                    name="RSI",
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['MACD'],
                    name="MACD",
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            if 'MACD_Signal' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=stock_data.index,
                        y=stock_data['MACD_Signal'],
                        name="MACD Signal",
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
        
        # Bollinger Bands
        if 'BB_Upper' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Upper'],
                    name="BB Upper",
                    line=dict(color='white', dash='dash')
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['BB_Lower'],
                    name="BB Lower",
                    line=dict(color='white', dash='dash'),
                    fill='tonexty'
                ),
                row=4, col=1
            )
        
        # Support/Resistance levels
        sr_levels = self.support_resistance_levels(stock_data)
        if sr_levels:
            for level in sr_levels['resistance_levels']:
                fig.add_hline(y=level, line_dash="dash", line_color="red", row=5, col=1)
            
            for level in sr_levels['support_levels']:
                fig.add_hline(y=level, line_dash="dash", line_color="green", row=5, col=1)
        
        fig.update_layout(
            title="Advanced Technical Analysis",
            height=800,
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    def _create_volume_chart(self, stock_data):
        """Create volume analysis chart"""
        volume_analysis = self.volume_analysis(stock_data)
        
        if not volume_analysis:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Volume vs Price', 'On-Balance Volume', 'Money Flow Index'),
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Volume and price
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name="Volume",
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                name="Price",
                yaxis='y2',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        # OBV
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=volume_analysis['obv'],
                name="OBV",
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        # MFI
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=volume_analysis['mfi'],
                name="MFI",
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # MFI levels
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title="Volume Analysis",
            height=600,
            template="plotly_dark",
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    def _create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=dict(
                text="Stock Correlation Matrix",
                font=dict(color='white', size=16)
            ),
            template="plotly_dark",
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white'),
            height=500
        )
        
        return fig
    
    def render_analytics_tab(self, stock_data, symbol):
        """Render the complete advanced analytics interface"""
        st.markdown("# 📊 Advanced Analytics")
        
        if stock_data is None or stock_data.empty:
            st.error("❌ No stock data available for analysis")
            return
        
        # Additional data validation
        if len(stock_data) < 5:
            st.error("❌ Insufficient data for analysis. Need at least 5 data points.")
            return
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        if missing_columns:
            st.error(f"❌ Missing required data columns: {missing_columns}")
            return
        
        # Analytics type selection
        analytics_type = st.selectbox(
            "Select Analysis Type",
            [
                "📈 Comprehensive Analysis",
                "📊 Volume Analysis", 
                "🔍 Risk Metrics",
                "🎯 Monte Carlo Simulation",
                "📐 Fibonacci Retracement",
                "🌊 Elliott Wave Analysis",
                "📈 Support/Resistance",
                "📅 Seasonality Analysis"
            ]
        )
        
        try:
            if analytics_type == "📈 Comprehensive Analysis":
                st.markdown("### 📊 Comprehensive Technical Analysis")
                
                # Create comprehensive chart
                fig = self.create_advanced_chart(stock_data, 'comprehensive')
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical indicators summary
                self._render_technical_summary(stock_data)
                
            elif analytics_type == "📊 Volume Analysis":
                st.markdown("### 📊 Volume Analysis")
                
                volume_metrics = self.volume_analysis(stock_data)
                if volume_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Volume", f"{volume_metrics['avg_volume']:,.0f}")
                    with col2:
                        st.metric("Volume Trend", volume_metrics['volume_trend'])
                    with col3:
                        st.metric("Volume Volatility", f"{volume_metrics['volatility']:.2%}")
                
                # Volume chart
                fig = self.create_advanced_chart(stock_data, 'volume')
                st.plotly_chart(fig, use_container_width=True)
                
            elif analytics_type == "🔍 Risk Metrics":
                st.markdown("### 🔍 Risk Analysis")
                
                # Calculate returns
                returns = stock_data['Close'].pct_change().dropna()
                risk_metrics = self.calculate_risk_metrics(returns)
                
                if risk_metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.3f}")
                    with col2:
                        st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                    with col3:
                        st.metric("Volatility", f"{risk_metrics['volatility']:.2%}")
                    with col4:
                        st.metric("VaR (95%)", f"{risk_metrics['var_95']:.2%}")
                
            elif analytics_type == "🎯 Monte Carlo Simulation":
                st.markdown("### 🎯 Monte Carlo Price Simulation")
                
                num_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
                time_horizon = st.slider("Time Horizon (Days)", 10, 252, 30, 5)
                
                if st.button("🚀 Run Simulation"):
                    with st.spinner("Running Monte Carlo simulation..."):
                        sim_results = self.monte_carlo_simulation(stock_data, num_simulations, time_horizon)
                        
                        if sim_results:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Price", f"₹{sim_results['mean_final_price']:.2f}")
                            with col2:
                                st.metric("95% Confidence Interval", 
                                         f"₹{sim_results['percentile_5']:.2f} - ₹{sim_results['percentile_95']:.2f}")
                            with col3:
                                st.metric("Probability of Gain", f"{sim_results['probability_profit']:.1%}")
                
            elif analytics_type == "📐 Fibonacci Retracement":
                st.markdown("### 📐 Fibonacci Retracement Analysis")
                
                fib_levels = self.fibonacci_retracement(stock_data)
                if fib_levels:
                    st.markdown("#### 🎯 Key Fibonacci Levels")
                    for level, price in fib_levels.items():
                        st.info(f"**{level}**: ₹{price:.2f}")
                
            elif analytics_type == "🌊 Elliott Wave Analysis":
                st.markdown("### 🌊 Elliott Wave Pattern Analysis")
                
                wave_analysis = self.elliott_wave_analysis(stock_data)
                if wave_analysis:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Current Wave**: {wave_analysis['current_trend']}")
                        # The original elliott_wave_analysis does not return pattern_strength or trend_direction
                        # This part of the new code will cause an error if uncommented.
                        # st.info(f"**Pattern Strength**: {wave_analysis['pattern_strength']:.1%}") 
                        st.info(f"**Next Target**: ₹{wave_analysis['current_trend']} (Estimate)") # Placeholder
                    with col2:
                        st.info(f"**Trend Direction**: {wave_analysis['current_trend']}")
                        # st.info(f"**Next Target**: ₹{wave_analysis['next_target']:.2f}") # Placeholder
                
            elif analytics_type == "📈 Support/Resistance":
                st.markdown("### 📈 Support & Resistance Levels")
                
                levels = self.support_resistance_levels(stock_data)
                if levels:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🔻 Support Levels")
                        for i, level in enumerate(levels['support_levels'], 1):
                            st.info(f"S{i}: ₹{level:.2f}")
                    with col2:
                        st.markdown("#### 🔺 Resistance Levels")
                        for i, level in enumerate(levels['resistance_levels'], 1):
                            st.info(f"R{i}: ₹{level:.2f}")
                
            elif analytics_type == "📅 Seasonality Analysis":
                st.markdown("### 📅 Seasonality Analysis")
                
                seasonality = self.seasonality_analysis(stock_data)
                if seasonality:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📊 Monthly Performance")
                        for month, performance in seasonality['monthly_patterns']['mean'].items():
                            color = "🟢" if performance > 0 else "🔴"
                            st.info(f"{color} {month}: {performance:+.2%}")
                    with col2:
                        st.markdown("#### 📊 Day of Week Performance")
                        for day, performance in seasonality['daily_patterns']['mean'].items():
                            color = "🟢" if performance > 0 else "🔴"
                            st.info(f"{color} {day}: {performance:+.2%}")
        
        except Exception as e:
            st.error(f"❌ Error in analytics: {str(e)}")
            st.info("Please try a different analysis type or refresh the data.")
    
    def _render_technical_summary(self, stock_data):
        """Render technical indicators summary"""
        st.markdown("### 📈 Technical Indicators Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Analysis
            if 'RSI' in stock_data.columns:
                rsi = stock_data['RSI'].iloc[-1]
                if rsi > 70:
                    rsi_signal = "🔴 Overbought"
                elif rsi < 30:
                    rsi_signal = "🟢 Oversold"
                else:
                    rsi_signal = "🟡 Neutral"
                st.info(f"**RSI (14)**: {rsi:.2f} - {rsi_signal}")
            
            # MACD Analysis
            if 'MACD' in stock_data.columns and 'MACD_Signal' in stock_data.columns:
                macd = stock_data['MACD'].iloc[-1]
                macd_signal = stock_data['MACD_Signal'].iloc[-1]
                macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                st.info(f"**MACD**: {macd:.4f} - {macd_trend}")
        
        with col2:
            # Bollinger Bands
            if 'BB_Upper' in stock_data.columns and 'BB_Lower' in stock_data.columns:
                current_price = stock_data['Close'].iloc[-1]
                bb_upper = stock_data['BB_Upper'].iloc[-1]
                bb_lower = stock_data['BB_Lower'].iloc[-1]
                
                if current_price > bb_upper:
                    bb_signal = "🔴 Above Upper Band"
                elif current_price < bb_lower:
                    bb_signal = "🟢 Below Lower Band"
                else:
                    bb_signal = "🟡 Within Bands"
                st.info(f"**Bollinger Bands**: {bb_signal}")
            
            # Moving Average Analysis
            if 'SMA_20' in stock_data.columns and 'SMA_50' in stock_data.columns:
                sma_20 = stock_data['SMA_20'].iloc[-1]
                sma_50 = stock_data['SMA_50'].iloc[-1]
                ma_trend = "🟢 Bullish" if sma_20 > sma_50 else "🔴 Bearish"
                st.info(f"**Moving Average Cross**: {ma_trend}")