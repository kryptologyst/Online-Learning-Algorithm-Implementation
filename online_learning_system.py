"""
Advanced Online Learning Algorithm Implementation
Project 71: Enhanced with modern ML techniques, streaming capabilities, and comprehensive evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from river import linear_model, metrics, preprocessing, compose
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Data class to store model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    prediction_time: float

class OnlineLearningSystem:
    """
    Advanced Online Learning System with multiple algorithms and comprehensive evaluation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.setup_models()
        
    def setup_models(self):
        """Initialize various online learning models"""
        # Scikit-learn models
        self.models['sgd_log'] = SGDClassifier(
            loss='log_loss', 
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state
        )
        
        self.models['sgd_hinge'] = SGDClassifier(
            loss='hinge',
            learning_rate='adaptive',
            eta0=0.01,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state
        )
        
        self.models['passive_aggressive'] = PassiveAggressiveClassifier(
            max_iter=1,
            warm_start=True,
            random_state=self.random_state
        )
        
        # River models (specialized for streaming)
        self.models['river_logistic'] = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.LogisticRegression()
        )
        
        self.models['river_pa'] = compose.Pipeline(
            preprocessing.StandardScaler(),
            linear_model.PAClassifier()
        )
        
        # Initialize scalers for sklearn models
        for model_name in ['sgd_log', 'sgd_hinge', 'passive_aggressive']:
            self.scalers[model_name] = StandardScaler()
            
        # Initialize performance tracking
        for model_name in self.models.keys():
            self.performance_history[model_name] = {
                'batch_accuracies': [],
                'batch_times': [],
                'cumulative_accuracy': []
            }
    
    def generate_streaming_data(self, n_samples: int = 10000, n_features: int = 20, 
                              n_classes: int = 2, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic streaming data with concept drift"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=self.random_state
        )
        
        # Add concept drift in the second half
        drift_point = n_samples // 2
        X[drift_point:] += np.random.normal(0, 0.5, X[drift_point:].shape)
        
        return X, y
    
    def train_batch_sklearn(self, model_name: str, X_batch: np.ndarray, y_batch: np.ndarray, 
                           is_first_batch: bool = False) -> float:
        """Train sklearn models on a batch"""
        model = self.models[model_name]
        scaler = self.scalers[model_name]
        
        start_time = time.time()
        
        if is_first_batch:
            X_scaled = scaler.fit_transform(X_batch)
            model.partial_fit(X_scaled, y_batch, classes=np.unique(y_batch))
        else:
            X_scaled = scaler.transform(X_batch)
            model.partial_fit(X_scaled, y_batch)
        
        training_time = time.time() - start_time
        
        # Calculate batch accuracy
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y_batch, y_pred)
        
        return accuracy, training_time
    
    def train_batch_river(self, model_name: str, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Train River models on a batch"""
        model = self.models[model_name]
        start_time = time.time()
        
        accuracies = []
        for x, y in zip(X_batch, y_batch):
            # Convert numpy array to dict for River
            x_dict = {f'feature_{i}': float(val) for i, val in enumerate(x)}
            
            # Make prediction before learning (for accuracy calculation)
            try:
                y_pred = model.predict_one(x_dict)
                accuracies.append(1 if y_pred == y else 0)
            except:
                accuracies.append(0)  # First prediction might fail
            
            # Learn from the sample
            model.learn_one(x_dict, y)
        
        training_time = time.time() - start_time
        batch_accuracy = np.mean(accuracies) if accuracies else 0.0
        
        return batch_accuracy, training_time
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Comprehensive model evaluation"""
        start_time = time.time()
        
        if 'river' in model_name:
            # River model evaluation
            predictions = []
            for x in X_test:
                x_dict = {f'feature_{i}': float(val) for i, val in enumerate(x)}
                try:
                    pred = self.models[model_name].predict_one(x_dict)
                    predictions.append(pred)
                except:
                    predictions.append(0)  # Default prediction
            y_pred = np.array(predictions)
        else:
            # Sklearn model evaluation
            X_test_scaled = self.scalers[model_name].transform(X_test)
            y_pred = self.models[model_name].predict(X_test_scaled)
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=report['weighted avg']['precision'],
            recall=report['weighted avg']['recall'],
            f1_score=report['weighted avg']['f1-score'],
            training_time=0,  # Will be set separately
            prediction_time=prediction_time
        )
    
    def train_streaming(self, X: np.ndarray, y: np.ndarray, batch_size: int = 100, 
                       test_size: float = 0.2) -> Dict[str, Any]:
        """Train all models on streaming data"""
        logger.info("Starting streaming training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        n_batches = int(np.ceil(len(X_train) / batch_size))
        results = {}
        
        logger.info(f"Training on {len(X_train)} samples in {n_batches} batches")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # Train sklearn models
            for model_name in ['sgd_log', 'sgd_hinge', 'passive_aggressive']:
                accuracy, training_time = self.train_batch_sklearn(
                    model_name, X_batch, y_batch, is_first_batch=(i == 0)
                )
                self.performance_history[model_name]['batch_accuracies'].append(accuracy)
                self.performance_history[model_name]['batch_times'].append(training_time)
            
            # Train River models
            for model_name in ['river_logistic', 'river_pa']:
                accuracy, training_time = self.train_batch_river(model_name, X_batch, y_batch)
                self.performance_history[model_name]['batch_accuracies'].append(accuracy)
                self.performance_history[model_name]['batch_times'].append(training_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed batch {i + 1}/{n_batches}")
        
        # Final evaluation
        logger.info("Performing final evaluation...")
        for model_name in self.models.keys():
            performance = self.evaluate_model(model_name, X_test, y_test)
            results[model_name] = performance
            logger.info(f"{model_name}: Accuracy = {performance.accuracy:.4f}")
        
        return results, X_test, y_test
    
    def create_visualizations(self, results: Dict[str, ModelPerformance]):
        """Create comprehensive visualizations"""
        # Performance comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'Training Time per Batch', 
                          'Batch-wise Accuracy Evolution', 'Precision vs Recall'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name].accuracy for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='Final Accuracy'),
            row=1, col=1
        )
        
        # 2. Training time comparison
        avg_times = [np.mean(self.performance_history[name]['batch_times']) for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=avg_times, name='Avg Training Time'),
            row=1, col=2
        )
        
        # 3. Batch-wise accuracy evolution
        for model_name in model_names:
            batch_accs = self.performance_history[model_name]['batch_accuracies']
            fig.add_trace(
                go.Scatter(y=batch_accs, name=f'{model_name}', mode='lines'),
                row=2, col=1
            )
        
        # 4. Precision vs Recall
        precisions = [results[name].precision for name in model_names]
        recalls = [results[name].recall for name in model_names]
        
        fig.add_trace(
            go.Scatter(x=precisions, y=recalls, mode='markers+text', 
                      text=model_names, textposition="top center",
                      name='Precision vs Recall'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Online Learning Algorithm Performance Analysis")
        
        return fig
    
    def save_models(self, save_dir: str = "models"):
        """Save trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            if 'river' not in model_name:
                # Save sklearn models with their scalers
                model_data = {
                    'model': model,
                    'scaler': self.scalers[model_name]
                }
                joblib.dump(model_data, save_path / f"{model_name}.pkl")
            else:
                # Save River models (they have built-in serialization)
                joblib.dump(model, save_path / f"{model_name}.pkl")
        
        logger.info(f"Models saved to {save_path}")
    
    def generate_report(self, results: Dict[str, ModelPerformance]) -> str:
        """Generate comprehensive performance report"""
        report = "# Online Learning Algorithm Performance Report\n\n"
        
        # Summary table
        report += "## Model Performance Summary\n\n"
        report += "| Model | Accuracy | Precision | Recall | F1-Score | Prediction Time (s) |\n"
        report += "|-------|----------|-----------|--------|----------|--------------------|\n"
        
        for model_name, perf in results.items():
            report += f"| {model_name} | {perf.accuracy:.4f} | {perf.precision:.4f} | "
            report += f"{perf.recall:.4f} | {perf.f1_score:.4f} | {perf.prediction_time:.4f} |\n"
        
        # Best performing model
        best_model = max(results.keys(), key=lambda k: results[k].accuracy)
        report += f"\n## Best Performing Model: {best_model}\n"
        report += f"- Accuracy: {results[best_model].accuracy:.4f}\n"
        report += f"- F1-Score: {results[best_model].f1_score:.4f}\n"
        
        # Training insights
        report += "\n## Training Insights\n"
        report += "- All models were trained using online/incremental learning\n"
        report += "- Data included concept drift simulation\n"
        report += "- River models are specifically designed for streaming data\n"
        report += "- SGD models use adaptive learning rates for better convergence\n"
        
        return report

def main():
    """Main execution function"""
    logger.info("Initializing Online Learning System...")
    
    # Initialize system
    ols = OnlineLearningSystem(random_state=42)
    
    # Generate streaming data
    logger.info("Generating streaming dataset...")
    X, y = ols.generate_streaming_data(n_samples=5000, n_features=15)
    
    # Train models
    results, X_test, y_test = ols.train_streaming(X, y, batch_size=50)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    fig = ols.create_visualizations(results)
    fig.write_html("online_learning_performance.html")
    
    # Save models
    ols.save_models()
    
    # Generate report
    report = ols.generate_report(results)
    with open("performance_report.md", "w") as f:
        f.write(report)
    
    logger.info("Analysis complete! Check 'online_learning_performance.html' and 'performance_report.md'")
    
    return ols, results

if __name__ == "__main__":
    system, results = main()
