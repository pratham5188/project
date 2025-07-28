#!/usr/bin/env python3
"""
StockTrendAI Startup Script
Optimized for deployment platforms
"""

import os
import sys
import subprocess
import time

def setup_environment():
    """Setup deployment environment"""
    print("🚀 StockTrendAI - Deployment Startup")
    print("=" * 50)
    
    # Set environment variables for production
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    print("✅ Environment variables configured")

def preload_models():
    """Preload critical components to reduce startup time"""
    try:
        print("📊 Pre-loading critical components...")
        
        # Import core modules
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        print("✅ Core data science libraries loaded")
        
        # Pre-import application modules
        from utils.data_fetcher import DataFetcher
        from utils.advanced_analytics import AdvancedAnalytics
        print("✅ Application modules loaded")
        
        print("🎯 Startup optimization complete")
        
    except Exception as e:
        print(f"⚠️  Pre-loading warning: {e}")
        print("🔄 Continuing with standard startup...")

def start_application():
    """Start the Streamlit application"""
    print("🌟 Starting StockTrendAI application...")
    
    # Get port from environment (for cloud platforms)
    port = os.environ.get('PORT', '8501')
    
    # Start Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', str(port),
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"🚀 Launching on port {port}")
    subprocess.run(cmd)

if __name__ == "__main__":
    try:
        setup_environment()
        preload_models()
        start_application()
    except KeyboardInterrupt:
        print("\n👋 StockTrendAI shutdown complete")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        sys.exit(1)