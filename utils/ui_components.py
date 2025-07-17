import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import time

class UIComponents:
    """Advanced UI components for enhanced user experience"""
    
    @staticmethod
    def create_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
        """Create a beautiful metric card using Streamlit components"""
        import streamlit as st
        
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
        
        return None  # Return None since we're using Streamlit components directly
    
    @staticmethod
    def create_progress_bar(progress, color="primary", animated=True):
        """Create an animated progress bar"""
        animation_class = "progress-animated" if animated else ""
        return f"""
        <div class="progress-container">
            <div class="progress-bar {animation_class}">
                <div class="progress-fill progress-{color}" style="width: {progress}%"></div>
            </div>
            <div class="progress-text">{progress}%</div>
        </div>
        """
    
    @staticmethod
    def create_alert_card(message, alert_type="info", dismissible=True):
        """Create alert cards with different types"""
        icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "info": "ℹ️"
        }
        
        dismiss_html = """
        <button class="alert-dismiss" onclick="this.parentElement.style.display='none';">×</button>
        """ if dismissible else ""
        
        return f"""
        <div class="alert alert-{alert_type}">
            <div class="alert-icon">{icons.get(alert_type, "ℹ️")}</div>
            <div class="alert-message">{message}</div>
            {dismiss_html}
        </div>
        """
    
    @staticmethod
    def create_loading_spinner(text="Loading..."):
        """Create a loading spinner with text"""
        return f"""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">{text}</div>
        </div>
        """
    
    @staticmethod
    def create_tab_navigation(tabs, active_tab=0):
        """Create custom tab navigation"""
        tab_html = '<div class="tab-navigation">'
        
        for i, tab in enumerate(tabs):
            active_class = "tab-active" if i == active_tab else ""
            tab_html += f'<button class="tab-button {active_class}" onclick="selectTab({i})">{tab}</button>'
        
        tab_html += '</div>'
        return tab_html
    
    @staticmethod
    def create_tooltip(text, tooltip_text):
        """Create text with tooltip"""
        return f"""
        <div class="tooltip-container">
            <span class="tooltip-text">{text}</span>
            <div class="tooltip-content">{tooltip_text}</div>
        </div>
        """
    
    @staticmethod
    def create_badge(text, badge_type="default"):
        """Create a badge component"""
        return f'<span class="badge badge-{badge_type}">{text}</span>'
    
    @staticmethod
    def create_card_container(content, title=None, collapsible=False):
        """Create a card container"""
        card_id = f"card_{int(time.time())}"
        
        title_html = ""
        if title:
            collapse_button = """
            <button class="card-collapse-btn" onclick="toggleCard('{card_id}')">
                <span class="collapse-icon">▼</span>
            </button>
            """ if collapsible else ""
            
            title_html = f"""
            <div class="card-header">
                <div class="card-title">{title}</div>
                {collapse_button}
            </div>
            """
        
        return f"""
        <div class="card-container" id="{card_id}">
            {title_html}
            <div class="card-content">
                {content}
            </div>
        </div>
        """
    
    @staticmethod
    def create_advanced_filter_panel():
        """Create an advanced filter panel"""
        return """
        <div class="filter-panel">
            <div class="filter-header">
                <h3>🔍 Advanced Filters</h3>
                <button class="filter-reset">Reset All</button>
            </div>
            <div class="filter-content">
                <div class="filter-group">
                    <label>Time Range</label>
                    <div class="filter-options">
                        <button class="filter-option active">1D</button>
                        <button class="filter-option">1W</button>
                        <button class="filter-option">1M</button>
                        <button class="filter-option">3M</button>
                        <button class="filter-option">1Y</button>
                    </div>
                </div>
                <div class="filter-group">
                    <label>Market Cap</label>
                    <input type="range" class="filter-slider" min="0" max="100" value="50">
                </div>
                <div class="filter-group">
                    <label>Volatility</label>
                    <input type="range" class="filter-slider" min="0" max="100" value="30">
                </div>
            </div>
        </div>
        """
    
    @staticmethod
    def create_floating_action_button(icon, tooltip, action):
        """Create a floating action button"""
        return f"""
        <div class="fab-container">
            <button class="fab" onclick="{action}" title="{tooltip}">
                <span class="fab-icon">{icon}</span>
            </button>
        </div>
        """
    
    @staticmethod
    def create_market_status_indicator(status="open"):
        """Create market status indicator"""
        status_colors = {
            "open": {"color": "green", "text": "Market Open"},
            "closed": {"color": "red", "text": "Market Closed"},
            "pre-market": {"color": "orange", "text": "Pre-Market"},
            "after-hours": {"color": "orange", "text": "After Hours"}
        }
        
        current_status = status_colors.get(status, status_colors["closed"])
        
        return f"""
        <div class="market-status">
            <div class="status-indicator status-{current_status['color']}"></div>
            <span class="status-text">{current_status['text']}</span>
        </div>
        """
    
    @staticmethod
    def create_comparison_chart(data1, data2, labels):
        """Create a comparison chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data1.index,
            y=data1.values,
            mode='lines',
            name=labels[0],
            line=dict(color='#00ff88', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=data2.index,
            y=data2.values,
            mode='lines',
            name=labels[1],
            line=dict(color='#ff0044', width=2)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            showlegend=True,
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(data, title="Heatmap"):
        """Create a heatmap visualization"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdBu',
            zmid=0,
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=400,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    @staticmethod
    def create_gauge_chart(value, title, min_val=0, max_val=100):
        """Create a gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [0, 30], 'color': "#ff0044"},
                    {'range': [30, 70], 'color': "#ffaa00"},
                    {'range': [70, 100], 'color': "#00ff88"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    @staticmethod
    def create_candlestick_chart(data, title="Price Chart"):
        """Create an enhanced candlestick chart"""
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff0044',
            increasing_fillcolor='rgba(0, 255, 136, 0.3)',
            decreasing_fillcolor='rgba(255, 0, 68, 0.3)'
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            yaxis='y2',
            opacity=0.3,
            marker_color='#888888'
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_dark",
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying='y', side='right'),
            xaxis_rangeslider_visible=False,
            height=500,
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    @staticmethod
    def create_performance_grid(performance_data):
        """Create a performance metrics grid"""
        grid_html = '<div class="performance-grid">'
        
        for metric, value in performance_data.items():
            # Handle string values that can't be compared numerically
            try:
                # Try to extract numeric value from string (e.g., "72.5%" -> 72.5)
                if isinstance(value, str):
                    # Extract numeric part from percentage strings
                    numeric_part = ''.join(c for c in value if c.isdigit() or c == '.' or c == '-')
                    if numeric_part:
                        numeric_value = float(numeric_part)
                        color_class = "positive" if numeric_value > 50 else "negative" if numeric_value < 30 else "neutral"
                    else:
                        color_class = "neutral"
                else:
                    color_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            except (ValueError, TypeError):
                color_class = "neutral"
            
            grid_html += f"""
            <div class="performance-item {color_class}">
                <div class="performance-label">{metric}</div>
                <div class="performance-value">{value}</div>
            </div>
            """
        
        grid_html += '</div>'
        return grid_html
    
    @staticmethod
    def create_news_feed(news_items):
        """Create a news feed component using Streamlit components"""
        import streamlit as st
        
        for item in news_items:
            sentiment = item.get('sentiment', {}).get('classification', 'neutral')
            sentiment_color = {"positive": "green", "negative": "red", "neutral": "blue"}.get(sentiment, "blue")
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{item.get('headline', 'No headline')}**")
                    st.write(item.get('summary', ''))
                    st.caption(f"Source: {item.get('source', 'Unknown')}")
                with col2:
                    st.write(f"**Sentiment:** :{sentiment_color}[{sentiment.title()}]")
                    st.caption(item.get('published_date', ''))
                st.divider()
        
        return None  # Return None since we're using Streamlit components directly
    
    @staticmethod
    def create_portfolio_summary(portfolio_data):
        """Create portfolio summary component"""
        if not portfolio_data:
            return '<div class="portfolio-summary">No portfolio data available</div>'
        
        total_value = portfolio_data.get('current_value', 0)
        total_invested = portfolio_data.get('total_invested', 0)
        profit_loss = portfolio_data.get('profit_loss', 0)
        profit_loss_percent = (profit_loss / total_invested * 100) if total_invested > 0 else 0
        
        color_class = "positive" if profit_loss > 0 else "negative" if profit_loss < 0 else "neutral"
        
        return f"""
        <div class="portfolio-summary">
            <div class="portfolio-header">
                <h3>📊 Portfolio Summary</h3>
            </div>
            <div class="portfolio-metrics">
                <div class="portfolio-metric">
                    <div class="metric-label">Total Value</div>
                    <div class="metric-value">₹{total_value:,.2f}</div>
                </div>
                <div class="portfolio-metric">
                    <div class="metric-label">Total Invested</div>
                    <div class="metric-value">₹{total_invested:,.2f}</div>
                </div>
                <div class="portfolio-metric {color_class}">
                    <div class="metric-label">P&L</div>
                    <div class="metric-value">₹{profit_loss:,.2f} ({profit_loss_percent:+.2f}%)</div>
                </div>
            </div>
        </div>
        """