"""
Auto-updater for stock discovery system
Runs periodic checks for new companies and updates the system automatically
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAutoUpdater:
    """Background service to automatically update stock lists"""
    
    def __init__(self, check_interval_hours: int = 24):
        self.check_interval = check_interval_hours * 3600  # Convert to seconds
        self.last_check = None
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
    
    def start(self):
        """Start the auto-updater in background"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_updater, daemon=True)
            self.thread.start()
            logger.info("Stock auto-updater started")
    
    def stop(self):
        """Stop the auto-updater"""
        if self.is_running:
            self.stop_event.set()
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=5)
            logger.info("Stock auto-updater stopped")
    
    def _run_updater(self):
        """Main updater loop"""
        while not self.stop_event.is_set():
            try:
                if self._should_check():
                    self._perform_update()
                    self.last_check = datetime.now()
                
                # Sleep for 1 hour intervals, checking stop event
                for _ in range(3600):  # 1 hour in seconds
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in auto-updater: {e}")
                # Sleep for 10 minutes on error before retrying
                for _ in range(600):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
    
    def _should_check(self) -> bool:
        """Check if it's time for an update"""
        if self.last_check is None:
            return True
        
        time_since_last = datetime.now() - self.last_check
        return time_since_last.total_seconds() >= self.check_interval
    
    def _perform_update(self):
        """Perform the actual stock update"""
        try:
            from utils.stock_discovery import auto_update_stocks
            
            logger.info("Performing scheduled stock list update...")
            success, message, count = auto_update_stocks()
            
            if success and count > 0:
                logger.info(f"Auto-update successful: {message}")
                # Store update info in session state for UI notification
                if 'auto_update_notifications' not in st.session_state:
                    st.session_state.auto_update_notifications = []
                
                notification = {
                    'timestamp': datetime.now().isoformat(),
                    'message': f"🆕 Auto-discovered {count} new companies",
                    'type': 'success'
                }
                st.session_state.auto_update_notifications.append(notification)
                
                # Keep only last 5 notifications
                if len(st.session_state.auto_update_notifications) > 5:
                    st.session_state.auto_update_notifications = st.session_state.auto_update_notifications[-5:]
            
            elif success:
                logger.info("Auto-update completed: No new stocks found")
            else:
                logger.warning(f"Auto-update failed: {message}")
                
        except Exception as e:
            logger.error(f"Failed to perform auto-update: {e}")
    
    def force_update(self) -> dict:
        """Force an immediate update and return result"""
        try:
            from utils.stock_discovery import auto_update_stocks
            
            logger.info("Performing forced stock list update...")
            success, message, count = auto_update_stocks()
            self.last_check = datetime.now()
            
            return {
                'success': success,
                'message': message,
                'new_stocks_count': count,
                'timestamp': self.last_check.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Forced update failed: {e}")
            return {
                'success': False,
                'message': f"Update failed: {str(e)}",
                'new_stocks_count': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status(self) -> dict:
        """Get current status of the auto-updater"""
        next_check = None
        if self.last_check:
            next_check_time = self.last_check + timedelta(seconds=self.check_interval)
            next_check = next_check_time.isoformat()
        
        return {
            'is_running': self.is_running,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'next_check': next_check,
            'check_interval_hours': self.check_interval / 3600
        }

# Global instance
auto_updater = StockAutoUpdater()

def start_auto_updater(check_interval_hours: int = 24):
    """Start the auto-updater service"""
    global auto_updater
    auto_updater.check_interval = check_interval_hours * 3600
    auto_updater.start()

def stop_auto_updater():
    """Stop the auto-updater service"""
    global auto_updater
    auto_updater.stop()

def force_stock_update() -> dict:
    """Force an immediate stock update"""
    global auto_updater
    return auto_updater.force_update()

def get_updater_status() -> dict:
    """Get current auto-updater status"""
    global auto_updater
    return auto_updater.get_status()

def get_update_notifications() -> list:
    """Get recent update notifications"""
    return st.session_state.get('auto_update_notifications', [])

def clear_notifications():
    """Clear update notifications"""
    if 'auto_update_notifications' in st.session_state:
        st.session_state.auto_update_notifications = []