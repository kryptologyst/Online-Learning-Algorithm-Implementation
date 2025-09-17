"""
Streamlit UI for Online Learning Algorithm System
Interactive web interface for real-time model training and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
from online_learning_system import OnlineLearningSystem
from mock_database import DataStreamManager
import logging

# Configure page
st.set_page_config(
    page_title="Online Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
    }
    .status-stopped {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ols' not in st.session_state:
    st.session_state.ols = None
if 'stream_manager' not in st.session_state:
    st.session_state.stream_manager = None
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = []

def initialize_system():
    """Initialize the online learning system"""
    if st.session_state.ols is None:
        with st.spinner("Initializing Online Learning System..."):
            st.session_state.ols = OnlineLearningSystem(random_state=42)
            st.session_state.stream_manager = DataStreamManager()
            st.success("System initialized successfully!")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Online Learning Algorithm Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Control Panel")
    
    # Initialize system
    if st.sidebar.button("Initialize System", type="primary"):
        initialize_system()
    
    if st.session_state.ols is None:
        st.info("👆 Please initialize the system using the sidebar button")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔄 Real-time Training", 
        "📈 Performance Analysis", 
        "🗄️ Data Management", 
        "⚙️ Model Configuration"
    ])
    
    with tab1:
        dashboard_tab()
    
    with tab2:
        training_tab()
    
    with tab3:
        analysis_tab()
    
    with tab4:
        data_tab()
    
    with tab5:
        config_tab()

def dashboard_tab():
    """Main dashboard overview"""
    st.header("System Overview")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Status", 
                 "🟢 Active" if st.session_state.ols else "🔴 Inactive")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        training_status = "🟢 Running" if st.session_state.training_active else "🔴 Stopped"
        st.metric("Training Status", training_status)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        model_count = len(st.session_state.ols.models) if st.session_state.ols else 0
        st.metric("Active Models", model_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        data_points = len(st.session_state.performance_data)
        st.metric("Data Points Processed", data_points)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick performance overview
    if st.session_state.results:
        st.subheader("Latest Model Performance")
        
        # Create performance comparison chart
        model_names = list(st.session_state.results.keys())
        accuracies = [st.session_state.results[name].accuracy for name in model_names]
        
        fig = go.Figure(data=[
            go.Bar(x=model_names, y=accuracies, 
                  marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ])
        
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def training_tab():
    """Real-time training interface"""
    st.header("Real-time Model Training")
    
    # Training configuration
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Configuration")
        
        batch_size = st.slider("Batch Size", 10, 200, 50)
        n_samples = st.slider("Total Samples", 1000, 10000, 5000)
        n_features = st.slider("Number of Features", 5, 30, 15)
        
        # Data source selection
        data_source = st.selectbox("Data Source", [
            "Synthetic", "Financial Stream", "IoT Sensors", "Mixed Sources"
        ])
        
        # Training controls
        if st.button("Start Training", type="primary"):
            start_training(batch_size, n_samples, n_features, data_source)
        
        if st.button("Stop Training", type="secondary"):
            st.session_state.training_active = False
            st.success("Training stopped")
    
    with col2:
        st.subheader("Training Progress")
        
        if st.session_state.training_active:
            # Create placeholder for real-time updates
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Simulate real-time training updates
            simulate_training_progress(progress_placeholder, chart_placeholder)
        else:
            st.info("Click 'Start Training' to begin real-time model training")

def analysis_tab():
    """Performance analysis and visualization"""
    st.header("Performance Analysis")
    
    if not st.session_state.results:
        st.info("No training results available. Please run training first.")
        return
    
    # Model comparison
    st.subheader("Detailed Model Comparison")
    
    # Create comprehensive comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    model_names = list(st.session_state.results.keys())
    
    # Add metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (metric, pos) in enumerate(zip(metrics, positions)):
        values = [getattr(st.session_state.results[name], metric) for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=values, name=metric.title(),
                  marker_color=f'rgba({31 + i*50}, {119 + i*30}, {180 + i*20}, 0.8)'),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.subheader("Performance Metrics Table")
    
    performance_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': f"{perf.accuracy:.4f}",
            'Precision': f"{perf.precision:.4f}",
            'Recall': f"{perf.recall:.4f}",
            'F1-Score': f"{perf.f1_score:.4f}",
            'Prediction Time (s)': f"{perf.prediction_time:.4f}"
        }
        for name, perf in st.session_state.results.items()
    ])
    
    st.dataframe(performance_df, use_container_width=True)
    
    # Best model highlight
    best_model = max(st.session_state.results.keys(), 
                    key=lambda k: st.session_state.results[k].accuracy)
    
    st.success(f"🏆 Best Performing Model: **{best_model}** "
              f"(Accuracy: {st.session_state.results[best_model].accuracy:.4f})")

def data_tab():
    """Data management and streaming"""
    st.header("Data Management")
    
    if st.session_state.stream_manager is None:
        st.warning("Stream manager not initialized")
        return
    
    # Data statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Statistics")
        
        if st.button("Refresh Statistics"):
            stats = st.session_state.stream_manager.db.get_data_statistics()
            
            st.metric("Total Records", stats['total_records'])
            
            if stats['label_distribution']:
                st.write("**Label Distribution:**")
                for label, count in stats['label_distribution'].items():
                    st.write(f"- Label {label}: {count} samples")
    
    with col2:
        st.subheader("Data Stream Control")
        
        stream_duration = st.slider("Stream Duration (minutes)", 1, 10, 2)
        
        if st.button("Start Data Stream"):
            with st.spinner("Starting data stream..."):
                st.session_state.stream_manager.db.start_background_streaming(stream_duration)
                st.success(f"Data stream started for {stream_duration} minutes")
        
        if st.button("Stop Data Stream"):
            st.session_state.stream_manager.db.stop_streaming()
            st.success("Data stream stopped")
    
    # Data export
    st.subheader("Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export to CSV"):
            st.session_state.stream_manager.db.export_data("exported_data.csv")
            st.success("Data exported to CSV")
    
    with col2:
        if st.button("Clear Database"):
            st.session_state.stream_manager.db.clear_data()
            st.success("Database cleared")
    
    with col3:
        if st.button("Populate Sample Data"):
            with st.spinner("Populating sample data..."):
                st.session_state.stream_manager.populate_initial_data(1000)
                st.success("Sample data populated")

def config_tab():
    """Model configuration and settings"""
    st.header("Model Configuration")
    
    if st.session_state.ols is None:
        st.warning("Online Learning System not initialized")
        return
    
    # Model selection
    st.subheader("Available Models")
    
    model_info = {
        'sgd_log': 'SGD with Logistic Loss - Fast, good for linearly separable data',
        'sgd_hinge': 'SGD with Hinge Loss - SVM-like, good for margin-based classification',
        'passive_aggressive': 'Passive Aggressive - Adaptive, good for noisy data',
        'river_logistic': 'River Logistic Regression - Optimized for streaming',
        'river_pa': 'River Passive Aggressive - Streaming-optimized PA classifier'
    }
    
    for model_name, description in model_info.items():
        with st.expander(f"📊 {model_name}"):
            st.write(description)
            
            if model_name in st.session_state.ols.performance_history:
                history = st.session_state.ols.performance_history[model_name]
                if history['batch_accuracies']:
                    st.write(f"**Latest Accuracy:** {history['batch_accuracies'][-1]:.4f}")
                    st.write(f"**Batches Processed:** {len(history['batch_accuracies'])}")
    
    # System settings
    st.subheader("System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Logging Level", 
                                ["INFO", "DEBUG", "WARNING", "ERROR"])
        
        auto_save = st.checkbox("Auto-save Models", value=True)
    
    with col2:
        max_memory = st.slider("Max Memory Usage (MB)", 100, 2000, 500)
        
        update_frequency = st.slider("UI Update Frequency (seconds)", 1, 10, 2)
    
    if st.button("Apply Settings"):
        st.success("Settings applied successfully!")

def start_training(batch_size, n_samples, n_features, data_source):
    """Start the training process"""
    st.session_state.training_active = True
    
    with st.spinner("Generating training data..."):
        # Generate data based on source
        if data_source == "Synthetic":
            X, y = st.session_state.ols.generate_streaming_data(n_samples, n_features)
        else:
            # For demo, use synthetic data with different parameters
            X, y = st.session_state.ols.generate_streaming_data(n_samples, n_features)
    
    with st.spinner("Training models..."):
        results, X_test, y_test = st.session_state.ols.train_streaming(
            X, y, batch_size=batch_size
        )
        
        st.session_state.results = results
        st.session_state.performance_data.extend([
            {'timestamp': datetime.now(), 'batch': i, 'accuracy': acc}
            for i, acc in enumerate(st.session_state.ols.performance_history['sgd_log']['batch_accuracies'])
        ])
    
    st.session_state.training_active = False
    st.success("Training completed successfully!")

def simulate_training_progress(progress_placeholder, chart_placeholder):
    """Simulate real-time training progress"""
    progress_bar = progress_placeholder.progress(0)
    
    # Simulate progress
    for i in range(100):
        progress_bar.progress(i + 1)
        
        # Update chart with simulated data
        if i % 10 == 0:
            # Create sample progress chart
            x_data = list(range(i + 1))
            y_data = [0.5 + 0.3 * np.sin(x/10) + np.random.normal(0, 0.05) for x in x_data]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers',
                                   name='Training Accuracy'))
            fig.update_layout(title="Real-time Training Progress",
                            xaxis_title="Batch", yaxis_title="Accuracy")
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.1)
        
        if not st.session_state.training_active:
            break

if __name__ == "__main__":
    main()
