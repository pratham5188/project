import streamlit as st

class ModelInfo:
    """Information and descriptions for all AI models in the application"""
    
    def __init__(self):
        self.models_info = {
            'XGBoost': {
                'icon': '🚀',
                'name': 'XGBoost (Speed)',
                'type': 'Gradient Boosting',
                'description': 'High-performance gradient boosting model optimized for speed and accuracy',
                'strengths': ['Fast prediction', 'Good with structured data', 'Feature importance'],
                'use_case': 'Quick market analysis and trend detection',
                'accuracy': 'Medium-High',
                'speed': 'Very Fast'
            },
            'LSTM': {
                'icon': '🧠',
                'name': 'LSTM (Deep Learning)',
                'type': 'Recurrent Neural Network',
                'description': 'Long Short-Term Memory network for capturing sequential patterns in stock data',
                'strengths': ['Sequential pattern recognition', 'Memory of past events', 'Complex relationships'],
                'use_case': 'Long-term trend analysis and pattern recognition',
                'accuracy': 'High',
                'speed': 'Medium'
            },
            'Prophet': {
                'icon': '📈',
                'name': 'Prophet (Time Series)',
                'type': 'Time Series Forecasting',
                'description': 'Facebook Prophet model specialized in time series forecasting with seasonality',
                'strengths': ['Seasonality detection', 'Trend analysis', 'Holiday effects'],
                'use_case': 'Seasonal trend prediction and long-term forecasting',
                'accuracy': 'High',
                'speed': 'Fast'
            },
            'Ensemble': {
                'icon': '🎯',
                'name': 'Ensemble (Multi-Model)',
                'type': 'Ensemble Learning',
                'description': 'Combines multiple machine learning algorithms for robust predictions',
                'strengths': ['Reduced overfitting', 'Higher accuracy', 'Multiple perspectives'],
                'use_case': 'Comprehensive analysis with multiple model consensus',
                'accuracy': 'Very High',
                'speed': 'Medium'
            },
            'Transformer': {
                'icon': '⚡',
                'name': 'Transformer (Attention)',
                'type': 'Attention Mechanism',
                'description': 'State-of-the-art attention-based model for complex pattern recognition',
                'strengths': ['Attention mechanism', 'Complex patterns', 'Market relationships'],
                'use_case': 'Advanced pattern recognition and market sentiment analysis',
                'accuracy': 'Very High',
                'speed': 'Medium-Slow'
            },
            'GRU': {
                'icon': '🔥',
                'name': 'GRU (Gated Recurrent Unit)',
                'type': 'Recurrent Neural Network',
                'description': 'GRU is a type of RNN that is efficient for sequential data and captures temporal dependencies with fewer parameters than LSTM.',
                'strengths': ['Efficient sequential modeling', 'Faster training than LSTM', 'Good for time series'],
                'use_case': 'Short- and medium-term trend prediction with efficient training',
                'accuracy': 'High',
                'speed': 'Fast-Medium'
            },
            'Stacking': {
                'icon': '🏆',
                'name': 'Stacking Ensemble',
                'type': 'Meta-Ensemble Learning',
                'description': 'Combines predictions from multiple base models using a meta-model for improved accuracy and robustness.',
                'strengths': ['Combines strengths of all models', 'Reduces bias and variance', 'Best overall performance'],
                'use_case': 'Ultimate consensus prediction for highest reliability',
                'accuracy': 'Very High',
                'speed': 'Medium'
            }
        }
    
    def get_model_info(self, model_name):
        """Get information for a specific model"""
        return self.models_info.get(model_name, {})
    
    def get_all_models(self):
        """Get all models information"""
        return self.models_info
    
    def render_model_comparison(self):
        """Render a comparison table of all models"""
        st.markdown("### 🤖 AI Models Comparison")
        
        # Create comparison table
        comparison_data = []
        for model_name, info in self.models_info.items():
            comparison_data.append([
                f"{info['icon']} {info['name']}",
                info['type'],
                info['accuracy'],
                info['speed'],
                info['use_case']
            ])
        
        import pandas as pd
        df = pd.DataFrame(comparison_data, columns=[
            'Model', 'Type', 'Accuracy', 'Speed', 'Best Use Case'
        ])
        
        st.dataframe(df, use_container_width=True)
    
    def render_model_details(self):
        """Render detailed information about each model"""
        st.markdown("### 📚 Detailed Model Information")
        
        for model_name, info in self.models_info.items():
            with st.expander(f"{info['icon']} {info['name']} - {info['type']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {info['description']}")
                    st.markdown("**Key Strengths:**")
                    for strength in info['strengths']:
                        st.markdown(f"• {strength}")
                    st.markdown(f"**Best Use Case:** {info['use_case']}")
                
                with col2:
                    st.metric("Accuracy", info['accuracy'])
                    st.metric("Speed", info['speed'])
    
    def render_model_recommendations(self):
        """Render recommendations for model selection"""
        st.markdown("### 💡 Model Selection Guide")
        
        recommendations = [
            {
                'scenario': '🚀 Quick Analysis',
                'models': ['XGBoost', 'GRU'],
                'reason': 'Fast predictions for quick market decisions'
            },
            {
                'scenario': '📊 Comprehensive Analysis',
                'models': ['Ensemble', 'Transformer', 'Stacking'],
                'reason': 'Multiple perspectives and meta-ensembling for thorough analysis'
            },
            {
                'scenario': '📈 Long-term Trends',
                'models': ['Prophet', 'LSTM', 'GRU'],
                'reason': 'Time series and sequential pattern expertise'
            },
            {
                'scenario': '🎯 Maximum Accuracy',
                'models': ['Ensemble', 'Transformer', 'LSTM', 'Stacking'],
                'reason': 'Advanced models and meta-ensembling for highest prediction accuracy'
            },
            {
                'scenario': '⚡ Real-time Trading',
                'models': ['XGBoost', 'Prophet', 'GRU'],
                'reason': 'Fast execution for time-sensitive decisions'
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"**{rec['scenario']}**")
            model_list = ', '.join([f"{self.models_info[m]['icon']} {m}" for m in rec['models']])
            st.markdown(f"• Recommended: {model_list}")
            st.markdown(f"• Reason: {rec['reason']}")
            st.markdown("---")
    
    def get_model_performance_badge(self, model_name, prediction_data):
        """Generate a performance badge for a model"""
        info = self.get_model_info(model_name)
        confidence = prediction_data.get('confidence', 50)
        
        # Determine badge color based on confidence
        if confidence >= 80:
            badge_color = "success"
        elif confidence >= 60:
            badge_color = "warning"
        else:
            badge_color = "error"
        
        return f"""
        <div class="model-badge badge-{badge_color}">
            {info['icon']} {model_name}
            <br>
            <small>{confidence:.1f}% confidence</small>
        </div>
        """
    
    def render_ensemble_explanation(self):
        """Explain how the ensemble model works"""
        st.markdown("### 🎯 How Ensemble Model Works")
        
        st.markdown("""
        The Ensemble model combines multiple machine learning algorithms to make more accurate predictions:
        
        **Components:**
        • 🌲 Random Forest - Tree-based ensemble for robust predictions
        • 📈 Gradient Boosting - Sequential model improvement
        • 📊 Logistic Regression - Linear relationship modeling  
        • 🔧 Support Vector Machine - Non-linear pattern recognition
        
        **Process:**
        1. Each component model makes individual predictions
        2. Predictions are weighted based on model performance
        3. Final prediction combines all model outputs
        4. Result is more stable and accurate than individual models
        
        **Benefits:**
        ✅ Reduced overfitting through model diversity
        ✅ Higher accuracy through consensus
        ✅ More robust to market volatility
        ✅ Better handling of different market conditions
        """)
    
    def render_transformer_explanation(self):
        """Explain how the transformer model works"""
        st.markdown("### ⚡ How Transformer Model Works")
        
        st.markdown("""
        The Transformer model uses attention mechanisms to understand complex market relationships:
        
        **Key Features:**
        • 🎯 **Multi-Head Attention** - Focuses on different aspects of data simultaneously
        • 🔄 **Self-Attention** - Understands relationships between different time periods
        • 📚 **Layer Normalization** - Stabilizes training and improves performance
        • 🎛️ **Feed-Forward Networks** - Processes attention outputs for predictions
        
        **Attention Mechanism:**
        The model automatically learns to focus on the most important:
        • Time periods (recent vs historical data)
        • Technical indicators (RSI, MACD, volume)
        • Price patterns (trends, reversals, volatility)
        • Market conditions (bull/bear markets)
        
        **Advantages:**
        ✅ Captures long-range dependencies in data
        ✅ Understands complex market relationships
        ✅ Adapts attention based on market conditions
        ✅ State-of-the-art performance in pattern recognition
        """)