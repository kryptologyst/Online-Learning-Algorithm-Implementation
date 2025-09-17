# Online Learning Algorithm Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

A comprehensive implementation of online learning algorithms with real-time streaming capabilities, interactive UI, and advanced performance monitoring.

## Features

- **Multiple Online Learning Algorithms**: SGD, Passive Aggressive, River-based streaming models
- **Real-time Data Streaming**: Mock database with simulated financial, IoT, and synthetic data streams
- **Interactive Web UI**: Streamlit-based dashboard for real-time monitoring and control
- **Comprehensive Evaluation**: Advanced metrics, visualizations, and performance tracking
- **Concept Drift Simulation**: Built-in support for concept drift in streaming data
- **Modern ML Stack**: Latest versions of scikit-learn, River, Plotly, and more

## Requirements

- Python 3.8+
- See `requirements.txt` for complete dependency list

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd 0071_Online_learning_algorithm_implementation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Online Learning System

Run the main online learning system:

```bash
python online_learning_system.py
```

This will:
- Initialize multiple online learning models
- Generate streaming data with concept drift
- Train models incrementally
- Generate performance reports and visualizations
- Save trained models

### 2. Interactive Web Dashboard

Launch the Streamlit UI for real-time interaction:

```bash
streamlit run streamlit_ui.py
```

Features include:
- Real-time model training visualization
- Performance monitoring dashboard
- Data stream management
- Model configuration interface

### 3. Mock Database Streaming

Test the streaming database functionality:

```bash
python mock_database.py
```

This demonstrates:
- Real-time data generation
- Multiple data source simulation
- Database operations for streaming data

## Architecture

### Core Components

1. **OnlineLearningSystem** (`online_learning_system.py`)
   - Multiple algorithm implementations
   - Batch processing for streaming data
   - Performance evaluation and visualization
   - Model persistence

2. **MockStreamingDatabase** (`mock_database.py`)
   - SQLite-based streaming data storage
   - Real-time data generation
   - Multiple data source simulation
   - Performance metrics storage

3. **Streamlit UI** (`streamlit_ui.py`)
   - Interactive web dashboard
   - Real-time monitoring
   - Configuration management
   - Data visualization

### Supported Algorithms

| Algorithm | Library | Best For |
|-----------|---------|----------|
| SGD Logistic | scikit-learn | General classification |
| SGD Hinge | scikit-learn | SVM-like classification |
| Passive Aggressive | scikit-learn | Noisy data streams |
| River Logistic | River | Optimized streaming |
| River PA | River | Streaming with adaptation |

## Performance Metrics

The system tracks comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Detailed performance metrics
- **Training Time**: Per-batch training duration
- **Prediction Time**: Inference latency
- **Memory Usage**: Resource consumption monitoring

## Data Streaming

### Supported Data Sources

1. **Synthetic Data**: Configurable classification datasets
2. **Financial Streams**: Simulated market data with price movements
3. **IoT Sensors**: Temperature, humidity, pressure readings
4. **Mixed Sources**: Combined multi-domain streaming

### Concept Drift Simulation

The system includes built-in concept drift simulation:
- Gradual drift in feature distributions
- Sudden changes in decision boundaries
- Adaptive learning rate adjustments

## Visualization

### Generated Outputs

1. **Performance Dashboard**: Interactive Plotly visualizations
2. **Training Progress**: Real-time accuracy tracking
3. **Model Comparison**: Side-by-side performance analysis
4. **Confusion Matrices**: Detailed classification results

### Export Formats

- HTML reports with interactive charts
- CSV data exports
- Markdown performance summaries
- Saved model artifacts

## Testing and Validation

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=.
```

### Code Quality

```bash
# Format code
black *.py

# Lint code
flake8 *.py
```

## 🔧 Configuration

### Model Parameters

Customize model behavior in `online_learning_system.py`:

```python
# SGD Configuration
SGDClassifier(
    loss='log_loss',
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=1,
    warm_start=True
)

# River Configuration
compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
)
```

### Streaming Parameters

Adjust streaming behavior in `mock_database.py`:

```python
# Stream configuration
stream_rate = 1.0  # samples per second
concept_drift_probability = 0.01  # drift likelihood
batch_size = 100  # processing batch size
```

## Usage Examples

### Example 1: Basic Training

```python
from online_learning_system import OnlineLearningSystem

# Initialize system
ols = OnlineLearningSystem(random_state=42)

# Generate streaming data
X, y = ols.generate_streaming_data(n_samples=5000)

# Train models
results, X_test, y_test = ols.train_streaming(X, y, batch_size=50)

# View results
for model_name, performance in results.items():
    print(f"{model_name}: {performance.accuracy:.4f}")
```

### Example 2: Real-time Streaming

```python
from mock_database import DataStreamManager

# Initialize stream manager
stream_manager = DataStreamManager()

# Start background streaming
stream_manager.db.start_background_streaming(duration_minutes=5)

# Process batches
for i in range(10):
    X_batch, y_batch = stream_manager.get_mixed_batch(100)
    # Process batch with your model
    print(f"Processed batch {i+1}: {len(X_batch)} samples")
```

### Example 3: Custom Data Source

```python
def create_custom_stream():
    """Create custom data stream"""
    features = np.random.randn(15)  # 15 features
    label = 1 if np.sum(features[:5]) > 0 else 0
    
    return {
        'features': features.tolist(),
        'label': label,
        'timestamp': datetime.now(),
        'data_source': 'custom_stream'
    }

# Use with stream manager
stream_manager.db.insert_streaming_data(create_custom_stream())
```

## Advanced Features

### 1. Model Ensemble

Combine multiple online learners:

```python
# Weighted voting ensemble
predictions = []
for model_name, model in ols.models.items():
    pred = model.predict(X_test_scaled)
    predictions.append(pred)

ensemble_pred = np.mean(predictions, axis=0) > 0.5
```

### 2. Adaptive Learning Rates

Implement dynamic learning rate adjustment:

```python
# Adaptive learning based on performance
if current_accuracy < threshold:
    model.set_params(eta0=model.eta0 * 1.1)  # Increase learning rate
else:
    model.set_params(eta0=model.eta0 * 0.95)  # Decrease learning rate
```

### 3. Online Feature Selection

Add feature selection for streaming data:

```python
from sklearn.feature_selection import SelectKBest, chi2

# Online feature selection
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X_batch, y_batch)
```

## Troubleshooting

### Common Issues

1. **Memory Usage**: Reduce batch size or enable model compression
2. **Slow Training**: Use River models for better streaming performance
3. **Poor Accuracy**: Check for concept drift and adjust learning rates
4. **Database Locks**: Ensure proper connection management in multi-threaded scenarios

### Performance Optimization

- Use appropriate batch sizes (50-200 samples)
- Enable warm_start for scikit-learn models
- Monitor memory usage with large datasets
- Use River models for high-throughput scenarios

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation in the code comments
- Review the example notebooks

## Future Enhancements

- [ ] Deep learning integration with PyTorch
- [ ] Distributed training with Dask
- [ ] Advanced concept drift detection
- [ ] Model interpretability features
- [ ] Production deployment templates
- [ ] A/B testing framework
- [ ] Real-time model monitoring
- [ ] Automated hyperparameter tuning


# Online-Learning-Algorithm-Implementation
