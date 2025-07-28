"""
Background Service for Automatic Stock Discovery
This service runs in the background to automatically discover new stocks and update the system
"""

import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundStockService:
    """Background service for automatic stock discovery and updates"""
    
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.last_update = None
        self.update_notifications = []
        self.notification_file = "data/notifications.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Load existing notifications
        self.load_notifications()
    
    def load_notifications(self):
        """Load notifications from file"""
        try:
            if os.path.exists(self.notification_file):
                with open(self.notification_file, 'r') as f:
                    data = json.load(f)
                    self.update_notifications = data.get('notifications', [])
                    self.last_update = data.get('last_update')
        except Exception as e:
            logger.error(f"Error loading notifications: {e}")
            self.update_notifications = []
    
    def save_notifications(self):
        """Save notifications to file"""
        try:
            data = {
                'notifications': self.update_notifications,
                'last_update': self.last_update
            }
            with open(self.notification_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving notifications: {e}")
    
    def add_notification(self, message: str, notification_type: str = "info"):
        """Add a new notification"""
        notification = {
            'message': message,
            'type': notification_type,
            'timestamp': datetime.now().isoformat(),
            'read': False
        }
        self.update_notifications.append(notification)
        
        # Keep only last 50 notifications
        if len(self.update_notifications) > 50:
            self.update_notifications = self.update_notifications[-50:]
        
        self.save_notifications()
        logger.info(f"Added notification: {message}")
    
    def get_unread_notifications(self) -> List[Dict]:
        """Get all unread notifications"""
        return [n for n in self.update_notifications if not n.get('read', False)]
    
    def mark_notifications_read(self):
        """Mark all notifications as read"""
        for notification in self.update_notifications:
            notification['read'] = True
        self.save_notifications()
    
    def clear_old_notifications(self):
        """Clear notifications older than 7 days"""
        cutoff_date = datetime.now() - timedelta(days=7)
        self.update_notifications = [
            n for n in self.update_notifications 
            if datetime.fromisoformat(n['timestamp']) > cutoff_date
        ]
        self.save_notifications()
    
    def check_for_new_stocks(self):
        """Check for new stocks and update the system"""
        try:
            logger.info("Starting automatic stock discovery check...")
            
            # Import here to avoid circular imports
            from utils.stock_discovery import auto_update_stocks
            
            # Run the auto-update
            result = auto_update_stocks()
            
            if result['success']:
                if result['new_stocks_count'] > 0:
                    message = f"🎉 Discovered {result['new_stocks_count']} new stocks! System updated automatically."
                    self.add_notification(message, "success")
                    logger.info(f"Found {result['new_stocks_count']} new stocks")
                else:
                    logger.info("No new stocks found - system is up to date")
                
                self.last_update = datetime.now().isoformat()
            else:
                error_msg = f"⚠️ Stock discovery failed: {result.get('error', 'Unknown error')}"
                self.add_notification(error_msg, "error")
                logger.error(f"Stock discovery failed: {result.get('error')}")
            
            # Clean up old notifications
            self.clear_old_notifications()
            
        except Exception as e:
            error_msg = f"⚠️ Error during stock discovery: {str(e)}"
            self.add_notification(error_msg, "error")
            logger.error(f"Error in stock discovery: {e}")
    
    def run_scheduled_tasks(self):
        """Run the scheduled task loop"""
        logger.info("Background stock service started")
        
        # Schedule tasks
        schedule.every(24).hours.do(self.check_for_new_stocks)  # Daily check
        schedule.every().monday.at("09:00").do(self.check_for_new_stocks)  # Weekly Monday check
        
        # Run initial check after 30 seconds
        schedule.every(30).seconds.do(self.check_for_new_stocks).tag('initial')
        
        while not self.stop_event.is_set():
            try:
                schedule.run_pending()
                
                # Cancel the initial check after first run
                schedule.clear('initial')
                
                # Sleep for 1 hour between checks
                self.stop_event.wait(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in scheduled task: {e}")
                self.stop_event.wait(60)  # Wait 1 minute before retrying
    
    def start(self):
        """Start the background service"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self.run_scheduled_tasks, daemon=True)
            self.thread.start()
            logger.info("Background stock service started successfully")
    
    def stop(self):
        """Stop the background service"""
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            schedule.clear()
            if self.thread:
                self.thread.join(timeout=5)
            logger.info("Background stock service stopped")
    
    def get_status(self) -> Dict:
        """Get service status"""
        return {
            'is_running': self.is_running,
            'last_update': self.last_update,
            'unread_notifications': len(self.get_unread_notifications()),
            'total_notifications': len(self.update_notifications)
        }

# Global service instance
_service = None

def get_service() -> BackgroundStockService:
    """Get the global service instance"""
    global _service
    if _service is None:
        _service = BackgroundStockService()
    return _service

def start_background_service():
    """Start the background service"""
    service = get_service()
    service.start()
    return service

def stop_background_service():
    """Stop the background service"""
    service = get_service()
    service.stop()

def get_notifications() -> List[Dict]:
    """Get all notifications"""
    service = get_service()
    return service.get_unread_notifications()

def mark_notifications_read():
    """Mark all notifications as read"""
    service = get_service()
    service.mark_notifications_read()

def get_service_status() -> Dict:
    """Get service status"""
    service = get_service()
    return service.get_status()

def manual_stock_check():
    """Manually trigger a stock check"""
    service = get_service()
    service.check_for_new_stocks()
    return "Stock check initiated"