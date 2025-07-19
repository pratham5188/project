import streamlit as st

class ModelInfo:
    """Information and descriptions for all AI models in the application"""
    
    def __init__(self):
        self.models_info = {
            'XGBoost': {
                'icon': '🚀',
                'name': 'XGBoost (Speed)',
                'type': 'Gradient Boosting',
                'description': 'XGBoost is a high-performance gradient boosting model optimized for speed and accuracy. It is widely used in structured/tabular data competitions and excels at handling large datasets with many features. XGBoost uses an ensemble of decision trees and advanced regularization to prevent overfitting.',
                'strengths': ['Fast prediction and training', 'Handles missing data well', 'Feature importance analysis', 'Robust to outliers', 'Scalable to large datasets'],
                'weaknesses': ['Not ideal for sequential/time series data', 'Requires careful hyperparameter tuning', 'Can overfit on small datasets'],
                'use_case': 'Quick market analysis, feature-driven predictions, and trend detection in structured data.',
                'limitations': 'Not suitable for capturing long-term dependencies or sequential patterns.',
                'example': 'Use XGBoost for rapid screening of stocks based on technical indicators and fundamentals.',
                'accuracy': 'Medium-High',
                'speed': 'Very Fast'
            },
            'LSTM': {
                'icon': '🧠',
                'name': 'LSTM (Deep Learning)',
                'type': 'Recurrent Neural Network',
                'description': 'LSTM (Long Short-Term Memory) networks are a type of RNN designed to capture long-term dependencies in sequential data. They are effective for time series forecasting, especially when past events influence future outcomes. LSTMs use memory cells and gates to control information flow.',
                'strengths': ['Captures long-term sequential patterns', 'Good for noisy time series', 'Handles variable-length sequences'],
                'weaknesses': ['Slower to train', 'Requires more data', 'Can be sensitive to hyperparameters'],
                'use_case': 'Long-term trend analysis, pattern recognition, and forecasting in stock prices.',
                'limitations': 'May overfit on small datasets; less interpretable than tree models.',
                'example': 'Use LSTM to predict stock prices based on historical price movements and volume.',
                'accuracy': 'High',
                'speed': 'Medium'
            },
            'Prophet': {
                'icon': '📈',
                'name': 'Prophet (Time Series)',
                'type': 'Time Series Forecasting',
                'description': 'Prophet is a time series forecasting model developed by Facebook. It is designed to handle seasonality, holidays, and trend changes automatically. Prophet is robust to missing data and outliers, making it suitable for business and financial forecasting.',
                'strengths': ['Automatic seasonality detection', 'Handles missing data', 'Easy to use and interpret', 'Good for business/financial data'],
                'weaknesses': ['Limited to additive models', 'Not ideal for highly non-linear patterns', 'Less flexible than deep learning models'],
                'use_case': 'Seasonal trend prediction, long-term forecasting, and business planning.',
                'limitations': 'Not suitable for high-frequency or highly volatile data.',
                'example': 'Use Prophet to forecast monthly or quarterly stock trends, accounting for holidays and seasonality.',
                'accuracy': 'High',
                'speed': 'Fast'
            },
            'Ensemble': {
                'icon': '🎯',
                'name': 'Ensemble (Multi-Model)',
                'type': 'Ensemble Learning',
                'description': 'The Ensemble model combines multiple machine learning algorithms (e.g., XGBoost, LSTM, Prophet) to produce more robust and accurate predictions. By aggregating the strengths of each model, ensemble methods reduce overfitting and improve generalization.',
                'strengths': ['Reduces overfitting', 'Higher accuracy', 'Combines multiple perspectives', 'Robust to noise'],
                'weaknesses': ['Slower inference', 'Complex to interpret', 'Requires more resources'],
                'use_case': 'Comprehensive analysis with consensus from multiple models.',
                'limitations': 'May be slower and harder to debug; requires all base models to be well-tuned.',
                'example': 'Use Ensemble for final buy/sell decisions by combining predictions from XGBoost, LSTM, and Prophet.',
                'accuracy': 'Very High',
                'speed': 'Medium'
            },
            'Transformer': {
                'icon': '⚡',
                'name': 'Transformer (Attention)',
                'type': 'Attention Mechanism',
                'description': 'Transformers use attention mechanisms to model complex relationships in sequential data. They are state-of-the-art for many NLP and time series tasks, capturing both short- and long-term dependencies. Transformers can process all time steps in parallel, making them efficient for large datasets.',
                'strengths': ['Captures complex patterns', 'Handles long-range dependencies', 'Parallel processing', 'Adaptable to various data types'],
                'weaknesses': ['Requires large datasets', 'High memory usage', 'Complex architecture'],
                'use_case': 'Advanced pattern recognition, market sentiment analysis, and multi-factor modeling.',
                'limitations': 'May overfit on small data; requires significant computational resources.',
                'example': 'Use Transformer to analyze multi-factor influences on stock prices, including news and technical indicators.',
                'accuracy': 'Very High',
                'speed': 'Medium-Slow'
            },
            'GRU': {
                'icon': '🔥',
                'name': 'GRU (Gated Recurrent Unit)',
                'type': 'Recurrent Neural Network',
                'description': 'GRU is a simplified version of LSTM that is efficient for sequential data and captures temporal dependencies with fewer parameters. It is faster to train and often performs similarly to LSTM on time series tasks.',
                'strengths': ['Efficient sequential modeling', 'Faster training than LSTM', 'Good for time series', 'Fewer parameters'],
                'weaknesses': ['May underperform on very complex sequences', 'Less expressive than LSTM'],
                'use_case': 'Short- and medium-term trend prediction with efficient training.',
                'limitations': 'Not as powerful as LSTM for highly complex patterns.',
                'example': 'Use GRU for quick, efficient stock price predictions on daily or weekly data.',
                'accuracy': 'High',
                'speed': 'Fast-Medium'
            },
            'Stacking': {
                'icon': '🏆',
                'name': 'Stacking Ensemble',
                'type': 'Meta-Ensemble Learning',
                'description': 'Stacking combines predictions from multiple base models using a meta-model (e.g., logistic regression) for improved accuracy and robustness. It leverages the strengths of all models and reduces both bias and variance.',
                'strengths': ['Combines strengths of all models', 'Reduces bias and variance', 'Best overall performance', 'Highly robust'],
                'weaknesses': ['Complex to implement', 'Requires careful validation', 'Slower inference'],
                'use_case': 'Ultimate consensus prediction for highest reliability and accuracy.',
                'limitations': 'Requires all base models to be well-tuned; can be slow to train and predict.',
                'example': 'Use Stacking for final portfolio allocation decisions, combining all model outputs.',
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
                    st.markdown("**Weaknesses:**")
                    for weakness in info.get('weaknesses', []):
                        st.markdown(f"• {weakness}")
                    st.markdown(f"**Best Use Case:** {info['use_case']}")
                    st.markdown(f"**Limitations:** {info['limitations']}")
                    st.markdown(f"**Example Scenario:** {info['example']}")
                
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