import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit as st

class PortfolioTracker:
    """Advanced portfolio tracking and management system"""
    
    def __init__(self):
        self.portfolio_key = "user_portfolio"
        self.watchlist_key = "user_watchlist"
        self.alerts_key = "user_alerts"
        
    def initialize_portfolio(self):
        """Initialize empty portfolio structure"""
        if self.portfolio_key not in st.session_state:
            st.session_state[self.portfolio_key] = {
                'holdings': [],
                'transactions': [],
                'total_invested': 0.0,
                'current_value': 0.0,
                'profit_loss': 0.0,
                'created_at': datetime.now().isoformat()
            }
    
    def get_portfolio(self):
        """Get current portfolio data"""
        self.initialize_portfolio()
        return st.session_state[self.portfolio_key]
    
    def add_holding(self, symbol, quantity, purchase_price, purchase_date=None):
        """Add a new holding to the portfolio"""
        self.initialize_portfolio()
        
        if purchase_date is None:
            purchase_date = datetime.now().isoformat()
        
        holding = {
            'symbol': symbol,
            'quantity': float(quantity),
            'purchase_price': float(purchase_price),
            'purchase_date': purchase_date,
            'current_price': float(purchase_price),
            'profit_loss': 0.0,
            'profit_loss_percent': 0.0
        }
        
        st.session_state[self.portfolio_key]['holdings'].append(holding)
        
        # Add transaction record
        transaction = {
            'type': 'BUY',
            'symbol': symbol,
            'quantity': float(quantity),
            'price': float(purchase_price),
            'date': purchase_date,
            'total_value': float(quantity) * float(purchase_price)
        }
        
        st.session_state[self.portfolio_key]['transactions'].append(transaction)
        self.update_portfolio_summary()
    
    def remove_holding(self, index):
        """Remove a holding from the portfolio"""
        if self.portfolio_key in st.session_state:
            holdings = st.session_state[self.portfolio_key]['holdings']
            if 0 <= index < len(holdings):
                removed_holding = holdings.pop(index)
                
                # Add sell transaction
                transaction = {
                    'type': 'SELL',
                    'symbol': removed_holding['symbol'],
                    'quantity': removed_holding['quantity'],
                    'price': removed_holding['current_price'],
                    'date': datetime.now().isoformat(),
                    'total_value': removed_holding['quantity'] * removed_holding['current_price']
                }
                
                st.session_state[self.portfolio_key]['transactions'].append(transaction)
                self.update_portfolio_summary()
    
    def update_portfolio_prices(self, data_fetcher):
        """Update current prices for all holdings"""
        if self.portfolio_key not in st.session_state:
            return
        
        holdings = st.session_state[self.portfolio_key]['holdings']
        
        for holding in holdings:
            try:
                # Get current price
                current_data = data_fetcher.get_stock_data(holding['symbol'], '1d')
                if current_data is not None and not current_data.empty:
                    current_price = float(current_data['Close'].iloc[-1])
                    holding['current_price'] = current_price
                    
                    # Calculate profit/loss
                    investment_value = holding['quantity'] * holding['purchase_price']
                    current_value = holding['quantity'] * current_price
                    profit_loss = current_value - investment_value
                    profit_loss_percent = (profit_loss / investment_value) * 100
                    
                    holding['profit_loss'] = profit_loss
                    holding['profit_loss_percent'] = profit_loss_percent
                    
            except Exception as e:
                print(f"Error updating price for {holding['symbol']}: {e}")
        
        self.update_portfolio_summary()
    
    def update_portfolio_summary(self):
        """Update portfolio summary statistics"""
        if self.portfolio_key not in st.session_state:
            return
        
        holdings = st.session_state[self.portfolio_key]['holdings']
        
        total_invested = sum(h['quantity'] * h['purchase_price'] for h in holdings)
        current_value = sum(h['quantity'] * h['current_price'] for h in holdings)
        profit_loss = current_value - total_invested
        
        st.session_state[self.portfolio_key]['total_invested'] = total_invested
        st.session_state[self.portfolio_key]['current_value'] = current_value
        st.session_state[self.portfolio_key]['profit_loss'] = profit_loss
    
    def get_portfolio_performance(self):
        """Get detailed portfolio performance metrics"""
        if self.portfolio_key not in st.session_state:
            return None
        
        portfolio = st.session_state[self.portfolio_key]
        holdings = portfolio['holdings']
        
        if not holdings:
            return None
        
        # Calculate various metrics
        total_invested = portfolio['total_invested']
        current_value = portfolio['current_value']
        profit_loss = portfolio['profit_loss']
        
        performance = {
            'total_invested': total_invested,
            'current_value': current_value,
            'profit_loss': profit_loss,
            'profit_loss_percent': (profit_loss / total_invested * 100) if total_invested > 0 else 0,
            'num_holdings': len(holdings),
            'best_performer': None,
            'worst_performer': None,
            'diversification_score': self.calculate_diversification_score(holdings)
        }
        
        # Find best and worst performers
        if holdings:
            best_performer = max(holdings, key=lambda x: x['profit_loss_percent'])
            worst_performer = min(holdings, key=lambda x: x['profit_loss_percent'])
            
            performance['best_performer'] = {
                'symbol': best_performer['symbol'],
                'profit_loss_percent': best_performer['profit_loss_percent'],
                'profit_loss': best_performer['profit_loss']
            }
            
            performance['worst_performer'] = {
                'symbol': worst_performer['symbol'],
                'profit_loss_percent': worst_performer['profit_loss_percent'],
                'profit_loss': worst_performer['profit_loss']
            }
        
        return performance
    
    def calculate_diversification_score(self, holdings):
        """Calculate portfolio diversification score (0-100)"""
        if not holdings:
            return 0
        
        # Simple diversification based on number of holdings and value distribution
        num_holdings = len(holdings)
        
        # Calculate weight distribution
        total_value = sum(h['quantity'] * h['current_price'] for h in holdings)
        if total_value == 0:
            return 0
        
        weights = [h['quantity'] * h['current_price'] / total_value for h in holdings]
        
        # Calculate concentration (higher concentration = lower diversification)
        concentration = sum(w**2 for w in weights)
        
        # Convert to diversification score (0-100)
        base_score = min(num_holdings * 10, 70)  # Max 70 points for number of holdings
        distribution_score = max(0, 30 - (concentration * 100))  # Max 30 points for distribution
        
        return min(100, base_score + distribution_score)
    
    def initialize_watchlist(self):
        """Initialize empty watchlist"""
        if self.watchlist_key not in st.session_state:
            st.session_state[self.watchlist_key] = []
    
    def add_to_watchlist(self, symbol, company_name):
        """Add stock to watchlist"""
        self.initialize_watchlist()
        
        # Check if already in watchlist
        existing = [item for item in st.session_state[self.watchlist_key] if item['symbol'] == symbol]
        if not existing:
            st.session_state[self.watchlist_key].append({
                'symbol': symbol,
                'company_name': company_name,
                'added_date': datetime.now().isoformat()
            })
    
    def remove_from_watchlist(self, symbol):
        """Remove stock from watchlist"""
        if self.watchlist_key in st.session_state:
            st.session_state[self.watchlist_key] = [
                item for item in st.session_state[self.watchlist_key] 
                if item['symbol'] != symbol
            ]
    
    def get_watchlist(self):
        """Get current watchlist"""
        self.initialize_watchlist()
        return st.session_state[self.watchlist_key]
    
    def initialize_alerts(self):
        """Initialize alerts system"""
        if self.alerts_key not in st.session_state:
            st.session_state[self.alerts_key] = []
    
    def add_price_alert(self, symbol, target_price, alert_type='above'):
        """Add price alert for a stock"""
        self.initialize_alerts()
        
        alert = {
            'id': len(st.session_state[self.alerts_key]),
            'symbol': symbol,
            'target_price': float(target_price),
            'alert_type': alert_type,  # 'above' or 'below'
            'created_date': datetime.now().isoformat(),
            'triggered': False,
            'triggered_date': None
        }
        
        st.session_state[self.alerts_key].append(alert)
    
    def check_alerts(self, data_fetcher):
        """Check and trigger price alerts"""
        if self.alerts_key not in st.session_state:
            return []
        
        triggered_alerts = []
        
        for alert in st.session_state[self.alerts_key]:
            if alert['triggered']:
                continue
            
            try:
                # Get current price
                current_data = data_fetcher.get_stock_data(alert['symbol'], '1d')
                if current_data is not None and not current_data.empty:
                    current_price = float(current_data['Close'].iloc[-1])
                    
                    # Check if alert should trigger
                    should_trigger = False
                    if alert['alert_type'] == 'above' and current_price >= alert['target_price']:
                        should_trigger = True
                    elif alert['alert_type'] == 'below' and current_price <= alert['target_price']:
                        should_trigger = True
                    
                    if should_trigger:
                        alert['triggered'] = True
                        alert['triggered_date'] = datetime.now().isoformat()
                        alert['triggered_price'] = current_price
                        triggered_alerts.append(alert)
                        
            except Exception as e:
                print(f"Error checking alert for {alert['symbol']}: {e}")
        
        return triggered_alerts
    
    def get_active_alerts(self):
        """Get all active (non-triggered) alerts"""
        self.initialize_alerts()
        return [alert for alert in st.session_state[self.alerts_key] if not alert['triggered']]
    
    def remove_alert(self, alert_id):
        """Remove an alert"""
        if self.alerts_key in st.session_state:
            st.session_state[self.alerts_key] = [
                alert for alert in st.session_state[self.alerts_key] 
                if alert['id'] != alert_id
            ]
    
    def render_portfolio_tab(self):
        """Render the complete portfolio management interface"""
        st.markdown("# 💼 Portfolio Tracker")
        
        # Initialize portfolio
        self.initialize_portfolio()
        
        # Create three columns for layout
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.markdown("### 📊 Portfolio Overview")
            
            portfolio = st.session_state[self.portfolio_key]
            holdings = portfolio.get('holdings', [])
            
            if holdings:
                # Update portfolio with current prices
                try:
                    from .data_fetcher import DataFetcher
                    data_fetcher = DataFetcher()
                    self.update_portfolio_prices(data_fetcher)
                    self.update_portfolio_summary()
                except Exception as e:
                    st.warning(f"⚠️ Could not update current prices: {str(e)}")
                
                # Display portfolio metrics
                total_invested = portfolio.get('total_invested', 0)
                current_value = portfolio.get('current_value', 0)
                profit_loss = portfolio.get('profit_loss', 0)
                
                # Metrics display
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("💰 Total Invested", f"₹{total_invested:,.2f}")
                with metric_col2:
                    st.metric("📈 Current Value", f"₹{current_value:,.2f}")
                with metric_col3:
                    profit_loss_percent = (profit_loss / total_invested * 100) if total_invested > 0 else 0
                    st.metric("📊 P&L", f"₹{profit_loss:,.2f}", f"{profit_loss_percent:+.2f}%")
                
                # Holdings table
                st.markdown("### 📋 Holdings")
                holdings_data = []
                for holding in holdings:
                    holdings_data.append({
                        'Symbol': holding['symbol'],
                        'Quantity': holding['quantity'],
                        'Purchase Price': f"₹{holding['purchase_price']:.2f}",
                        'Current Price': f"₹{holding['current_price']:.2f}",
                        'P&L': f"₹{holding['profit_loss']:.2f}",
                        'P&L %': f"{holding['profit_loss_percent']:+.2f}%"
                    })
                
                import pandas as pd
                df = pd.DataFrame(holdings_data)
                st.dataframe(df, use_container_width=True)
                
            else:
                st.info("📝 No holdings found. Add your first stock below!")
        
        with col2:
            st.markdown("### ➕ Add New Holding")
            
            with st.form("add_holding_form"):
                symbol_input = st.text_input("Stock Symbol (e.g., RELIANCE.NS)")
                quantity_input = st.number_input("Quantity", min_value=1, value=1)
                price_input = st.number_input("Purchase Price (₹)", min_value=0.01, value=100.0, step=0.01)
                purchase_date = st.date_input("Purchase Date")
                
                if st.form_submit_button("🚀 Add Holding"):
                    if symbol_input:
                        try:
                            self.add_holding(
                                symbol_input.upper(),
                                quantity_input,
                                price_input,
                                purchase_date.isoformat()
                            )
                            st.success(f"✅ Added {quantity_input} shares of {symbol_input} to portfolio!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error adding holding: {str(e)}")
                    else:
                        st.error("Please enter a stock symbol")
        
        with col3:
            st.markdown("### 🗑️ Manage Holdings")
            
            holdings = st.session_state[self.portfolio_key].get('holdings', [])
            if holdings:
                # Remove holding
                holding_options = [f"{h['symbol']} ({h['quantity']} shares)" for h in holdings]
                selected_holding = st.selectbox("Select holding to remove", holding_options)
                
                if st.button("🗑️ Remove", type="secondary"):
                    if selected_holding:
                        index = holding_options.index(selected_holding)
                        self.remove_holding(index)
                        st.success("✅ Holding removed!")
                        st.rerun()
            else:
                st.info("No holdings to manage")
        
        # Watchlist section
        st.markdown("---")
        st.markdown("### 👁️ Watchlist")
        
        watchlist_col1, watchlist_col2 = st.columns([2, 1])
        
        with watchlist_col1:
            watchlist = self.get_watchlist()
            if watchlist:
                watchlist_symbols = [item['symbol'] for item in watchlist]
                st.info(f"📊 Watching: {', '.join(watchlist_symbols)}")
            else:
                st.info("📝 No stocks in watchlist")
        
        with watchlist_col2:
            with st.form("add_watchlist_form"):
                watch_symbol = st.text_input("Add to Watchlist")
                if st.form_submit_button("👁️ Watch"):
                    if watch_symbol:
                        self.add_to_watchlist(watch_symbol.upper(), watch_symbol.upper())
                        st.success(f"✅ Added {watch_symbol} to watchlist!")
                        st.rerun()
        
        # Price alerts section
        st.markdown("### 🚨 Price Alerts")
        
        alert_col1, alert_col2 = st.columns([2, 1])
        
        with alert_col1:
            active_alerts = self.get_active_alerts()
            if active_alerts:
                for alert in active_alerts:
                    direction = "above" if alert['alert_type'] == 'above' else "below"
                    st.info(f"🚨 {alert['symbol']}: Alert when price goes {direction} ₹{alert['target_price']:.2f}")
            else:
                st.info("📝 No active alerts")
        
        with alert_col2:
            with st.form("add_alert_form"):
                alert_symbol = st.text_input("Symbol for Alert")
                target_price = st.number_input("Target Price (₹)", min_value=0.01, value=100.0)
                alert_type = st.selectbox("Alert Type", ["above", "below"])
                
                if st.form_submit_button("🚨 Set Alert"):
                    if alert_symbol:
                        self.add_price_alert(alert_symbol.upper(), target_price, alert_type)
                        st.success(f"✅ Alert set for {alert_symbol}!")
                        st.rerun()