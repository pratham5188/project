def get_custom_css():
    """Return custom CSS for futuristic 3D neon glow interface"""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Header */
    .neon-header {
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, rgba(0,255,136,0.1), rgba(0,136,255,0.1));
        border-radius: 20px;
        border: 2px solid rgba(0,255,136,0.3);
        box-shadow: 
            0 0 20px rgba(0,255,136,0.3),
            inset 0 0 20px rgba(0,255,136,0.1);
        animation: pulse-glow 3s ease-in-out infinite alternate;
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00ff88, #00aaff, #ff00aa);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 4s ease infinite;
        text-shadow: 0 0 30px rgba(0,255,136,0.5);
        margin: 0;
    }
    
    .subtitle {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        color: #00aaff;
        text-shadow: 0 0 10px rgba(0,170,255,0.5);
        margin-top: 1rem;
    }
    
    /* Sidebar Styles */
    .sidebar-header h2 {
        font-family: 'Orbitron', monospace;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
        text-align: center;
        border-bottom: 2px solid rgba(0,255,136,0.3);
        padding-bottom: 1rem;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(26,26,46,0.8));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    /* Advanced UI Components */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,136,255,0.1));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,255,136,0.3);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00ff88;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .metric-positive { color: #00ff88; }
    .metric-negative { color: #ff0044; }
    .metric-neutral { color: #ffffff; }
    
    /* Progress Bar */
    .progress-container {
        position: relative;
        margin: 1rem 0;
    }
    
    .progress-bar {
        width: 100%;
        height: 20px;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .progress-primary { background: linear-gradient(90deg, #00ff88, #0088ff); }
    .progress-success { background: linear-gradient(90deg, #00ff88, #88ff00); }
    .progress-warning { background: linear-gradient(90deg, #ffaa00, #ff8800); }
    .progress-danger { background: linear-gradient(90deg, #ff0044, #ff4400); }
    
    .progress-animated {
        animation: progress-shimmer 2s infinite;
    }
    
    @keyframes progress-shimmer {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .progress-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #fff;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Alert Cards */
    .alert {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        border: 1px solid;
        position: relative;
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,100,0.1));
        border-color: rgba(0,255,136,0.3);
        color: #00ff88;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,136,0,0.1));
        border-color: rgba(255,170,0,0.3);
        color: #ffaa00;
    }
    
    .alert-error {
        background: linear-gradient(135deg, rgba(255,0,68,0.1), rgba(200,0,50,0.1));
        border-color: rgba(255,0,68,0.3);
        color: #ff0044;
    }
    
    .alert-info {
        background: linear-gradient(135deg, rgba(0,136,255,0.1), rgba(0,100,200,0.1));
        border-color: rgba(0,136,255,0.3);
        color: #0088ff;
    }
    
    .alert-dismiss {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        background: none;
        border: none;
        color: inherit;
        font-size: 1.2rem;
        cursor: pointer;
        padding: 0;
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Loading Spinner */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(0,255,136,0.3);
        border-top: 4px solid #00ff88;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        margin-top: 1rem;
        color: #ffffff;
    }
    
    /* Card Container */
    .card-container {
        background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(26,26,46,0.8));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 15px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .card-header {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,136,255,0.1));
        padding: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(0,255,136,0.3);
    }
    
    .card-title {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        font-weight: bold;
        color: #00ff88;
    }
    
    .card-content {
        padding: 1rem;
    }
    
    /* Badge */
    .badge {
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .badge-default { background: rgba(136,136,136,0.3); color: #ffffff; }
    .badge-primary { background: linear-gradient(45deg, #00ff88, #0088ff); color: #fff; }
    .badge-success { background: linear-gradient(45deg, #00ff88, #88ff00); color: #fff; }
    .badge-warning { background: linear-gradient(45deg, #ffaa00, #ff8800); color: #fff; }
    .badge-danger { background: linear-gradient(45deg, #ff0044, #ff4400); color: #fff; }
    
    /* Portfolio Summary */
    .portfolio-summary {
        background: linear-gradient(135deg, rgba(0,136,255,0.1), rgba(136,0,255,0.1));
        border: 1px solid rgba(0,136,255,0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .portfolio-header h3 {
        font-family: 'Orbitron', monospace;
        color: #0088ff;
        margin-bottom: 1rem;
    }
    
    .portfolio-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
    }
    
    .portfolio-metric {
        text-align: center;
        padding: 1rem;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .portfolio-metric.positive {
        border-color: rgba(0,255,136,0.3);
    }
    
    .portfolio-metric.negative {
        border-color: rgba(255,0,68,0.3);
    }
    
    /* News Feed */
    .news-feed {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
    }
    
    .news-item {
        background: rgba(0,0,0,0.3);
        border-left: 4px solid rgba(136,136,136,0.5);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    
    .news-item.sentiment-positive {
        border-left-color: #00ff88;
    }
    
    .news-item.sentiment-negative {
        border-left-color: #ff0044;
    }
    
    .news-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .news-source {
        font-size: 0.8rem;
        color: #ffffff;
        font-weight: bold;
    }
    
    .news-time {
        font-size: 0.7rem;
        color: #ffffff;
    }
    
    .news-headline {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #fff;
    }
    
    .news-summary {
        font-size: 0.9rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .news-sentiment {
        font-size: 0.8rem;
        color: #ffffff;
    }
    
    /* Market Status */
    .market-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(0,0,0,0.3);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-green { background: #00ff88; }
    .status-red { background: #ff0044; }
    .status-orange { background: #ffaa00; }
    
    .status-text {
        font-size: 0.9rem;
        font-weight: bold;
        color: #fff;
    }
    
    /* Performance Grid */
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .performance-item {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .performance-item.positive {
        border-color: rgba(0,255,136,0.3);
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,100,0.1));
    }
    
    .performance-item.negative {
        border-color: rgba(255,0,68,0.3);
        background: linear-gradient(135deg, rgba(255,0,68,0.1), rgba(200,0,50,0.1));
    }
    
    .performance-label {
        font-size: 0.8rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .performance-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #fff;
    }
    
    /* Floating Action Button */
    .fab-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    .fab {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(45deg, #00ff88, #0088ff);
        border: none;
        color: #fff;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(0,255,136,0.3);
        transition: all 0.3s ease;
    }
    
    .fab:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 30px rgba(0,255,136,0.5);
    }
    
    /* Tab Navigation */
    .tab-navigation {
        display: flex;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .tab-button {
        flex: 1;
        padding: 0.7rem 1rem;
        background: transparent;
        border: none;
        color: #ffffff;
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .tab-button.tab-active {
        background: linear-gradient(45deg, #00ff88, #0088ff);
        color: #fff;
    }
    
    .tab-button:hover {
        color: #fff;
        background: rgba(0,255,136,0.2);
    }
    
    /* Tooltip */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-text {
        border-bottom: 1px dotted #ffffff;
        cursor: help;
    }
    
    .tooltip-content {
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0,0,0,0.9);
        color: #fff;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .tooltip-container:hover .tooltip-content {
        opacity: 1;
        visibility: visible;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .prediction-card:hover::before {
        transform: translateX(100%);
    }
    
    .prediction-card-up {
        border-color: rgba(0,255,136,0.5);
        box-shadow: 0 0 20px rgba(0,255,136,0.2);
    }
    
    .prediction-card-up:hover {
        box-shadow: 0 0 30px rgba(0,255,136,0.4);
        transform: translateY(-5px) scale(1.02);
    }
    
    .prediction-card-down {
        border-color: rgba(255,0,68,0.5);
        box-shadow: 0 0 20px rgba(255,0,68,0.2);
    }
    
    .prediction-card-down:hover {
        box-shadow: 0 0 30px rgba(255,0,68,0.4);
        transform: translateY(-5px) scale(1.02);
    }
    
    .model-name {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .prediction-direction {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 0 0 10px currentColor;
    }
    
    .confidence {
        font-family: 'Orbitron', monospace;
        font-size: 1rem;
        text-align: center;
        color: #ffaa00;
        text-shadow: 0 0 10px rgba(255,170,0,0.5);
    }
    
    .price-prediction {
        text-align: center;
        margin-top: 1rem;
        font-family: 'Orbitron', monospace;
    }
    
    .price-change {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    .current-price {
        font-size: 0.9rem;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    
    .predicted-price {
        font-size: 1.1rem;
        font-weight: bold;
        color: #00ff88;
        margin-bottom: 0.3rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,0,0,0.6), rgba(26,26,46,0.6));
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(0,255,136,0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,255,136,0.3);
    }
    
    .metric-title {
        font-size: 0.8rem;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0,255,136,0.3);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.1), transparent);
        border-top: 1px solid rgba(0,255,136,0.3);
        font-size: 0.9rem;
        color: #ffffff;
    }
    
    /* Animations */
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 20px rgba(0,255,136,0.3), inset 0 0 20px rgba(0,255,136,0.1); }
        100% { box-shadow: 0 0 40px rgba(0,255,136,0.5), inset 0 0 30px rgba(0,255,136,0.2); }
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .neon-header {
            padding: 1rem;
        }
        
        .prediction-card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 0.8rem;
        }
        
        .metric-value {
            font-size: 1.2rem;
        }
    }
    
    /* Custom Streamlit Component Styling */
    .stSelectbox > div > div {
        background-color: rgba(0,0,0,0.5);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 5px;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, rgba(0,255,136,0.2), rgba(0,136,255,0.2));
        border: 1px solid rgba(0,255,136,0.5);
        border-radius: 10px;
        color: #ffffff;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, rgba(0,255,136,0.4), rgba(0,136,255,0.4));
        box-shadow: 0 0 15px rgba(0,255,136,0.3);
        transform: translateY(-2px);
    }
    
    .stCheckbox > label {
        color: #ffffff;
        font-family: 'Orbitron', monospace;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(0,255,136,0.1);
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 5px;
    }
    
    .stError {
        background-color: rgba(255,0,68,0.1);
        border: 1px solid rgba(255,0,68,0.3);
        border-radius: 5px;
    }
    
    .stWarning {
        background-color: rgba(255,170,0,0.1);
        border: 1px solid rgba(255,170,0,0.3);
        border-radius: 5px;
    }
    
    .stInfo {
        background-color: rgba(0,136,255,0.1);
        border: 1px solid rgba(0,136,255,0.3);
        border-radius: 5px;
    }
    
    /* Plotly Chart Styling */
    .js-plotly-plot {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0,255,136,0.1);
        background-color: black !important;
    }
    
    /* Ensure all plotly elements have black background */
    .plotly .bg-white {
        background-color: black !important;
    }
    
    /* Plotly modebar styling */
    .modebar {
        background-color: black !important;
    }
    
    /* Plotly legend styling */
    .legend {
        background-color: black !important;
        color: white !important;
    }
    
    </style>
    """
