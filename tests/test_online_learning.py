"""
Test suite for Online Learning Algorithm Implementation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_learning_system import OnlineLearningSystem, ModelPerformance
from mock_database import MockStreamingDatabase, DataStreamManager

class TestOnlineLearningSystem:
    """Test cases for OnlineLearningSystem"""
    
    @pytest.fixture
    def ols(self):
        """Create OnlineLearningSystem instance for testing"""
        return OnlineLearningSystem(random_state=42)
    
    def test_initialization(self, ols):
        """Test system initialization"""
        assert len(ols.models) == 5
        assert 'sgd_log' in ols.models
        assert 'river_logistic' in ols.models
        assert len(ols.scalers) == 3  # Only sklearn models have scalers
    
    def test_data_generation(self, ols):
        """Test streaming data generation"""
        X, y = ols.generate_streaming_data(n_samples=100, n_features=10)
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert len(np.unique(y)) <= 2  # Binary classification
    
    def test_sklearn_batch_training(self, ols):
        """Test sklearn model batch training"""
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        accuracy, training_time = ols.train_batch_sklearn('sgd_log', X, y, is_first_batch=True)
        
        assert 0 <= accuracy <= 1
        assert training_time >= 0
    
    def test_river_batch_training(self, ols):
        """Test River model batch training"""
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        accuracy, training_time = ols.train_batch_river('river_logistic', X, y)
        
        assert 0 <= accuracy <= 1
        assert training_time >= 0
    
    def test_model_evaluation(self, ols):
        """Test model evaluation"""
        # Train a simple model first
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        ols.train_batch_sklearn('sgd_log', X_train, y_train, is_first_batch=True)
        
        # Evaluate
        X_test = np.random.randn(50, 10)
        y_test = np.random.randint(0, 2, 50)
        
        performance = ols.evaluate_model('sgd_log', X_test, y_test)
        
        assert isinstance(performance, ModelPerformance)
        assert 0 <= performance.accuracy <= 1
        assert 0 <= performance.precision <= 1
        assert 0 <= performance.recall <= 1
        assert 0 <= performance.f1_score <= 1
    
    def test_streaming_training(self, ols):
        """Test complete streaming training pipeline"""
        X, y = ols.generate_streaming_data(n_samples=200, n_features=5)
        
        results, X_test, y_test = ols.train_streaming(X, y, batch_size=50)
        
        assert len(results) == len(ols.models)
        assert X_test.shape[1] == 5
        assert len(X_test) > 0
        
        for model_name, performance in results.items():
            assert isinstance(performance, ModelPerformance)
            assert 0 <= performance.accuracy <= 1

class TestMockStreamingDatabase:
    """Test cases for MockStreamingDatabase"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db = MockStreamingDatabase(temp_file.name)
        yield db
        
        db.close()
        os.unlink(temp_file.name)
    
    def test_database_initialization(self, temp_db):
        """Test database setup"""
        cursor = temp_db.connection.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'streaming_data' in tables
        assert 'model_predictions' in tables
        assert 'model_performance' in tables
    
    def test_data_generation(self, temp_db):
        """Test synthetic data generation"""
        data_point = temp_db.generate_synthetic_stream(n_features=10)
        
        assert 'features' in data_point
        assert 'label' in data_point
        assert 'timestamp' in data_point
        assert len(data_point['features']) == 10
        assert data_point['label'] in [0, 1]
    
    def test_data_insertion_retrieval(self, temp_db):
        """Test data insertion and retrieval"""
        # Insert test data
        data_point = temp_db.generate_synthetic_stream(n_features=5)
        temp_db.insert_streaming_data(data_point, batch_id=1)
        
        # Retrieve data
        X, y = temp_db.get_batch_data(batch_size=10, batch_id=1)
        
        assert len(X) == 1
        assert len(y) == 1
        assert X.shape[1] == 5
    
    def test_prediction_storage(self, temp_db):
        """Test prediction storage"""
        features = [1.0, 2.0, 3.0]
        temp_db.store_prediction('test_model', 1, 0.8, 1, features)
        
        cursor = temp_db.connection.cursor()
        cursor.execute("SELECT * FROM model_predictions")
        rows = cursor.fetchall()
        
        assert len(rows) == 1
        assert rows[0][2] == 'test_model'  # model_name
        assert rows[0][3] == 1  # prediction
    
    def test_performance_storage(self, temp_db):
        """Test performance metrics storage"""
        temp_db.store_performance_metrics('test_model', 1, 0.85, 0.80, 0.90, 0.85)
        
        cursor = temp_db.connection.cursor()
        cursor.execute("SELECT * FROM model_performance")
        rows = cursor.fetchall()
        
        assert len(rows) == 1
        assert rows[0][2] == 'test_model'  # model_name
        assert rows[0][4] == 0.85  # accuracy
    
    def test_statistics(self, temp_db):
        """Test database statistics"""
        # Insert some test data
        for i in range(5):
            data_point = temp_db.generate_synthetic_stream()
            temp_db.insert_streaming_data(data_point)
        
        stats = temp_db.get_data_statistics()
        
        assert stats['total_records'] == 5
        assert 'label_distribution' in stats
        assert 'source_distribution' in stats

class TestDataStreamManager:
    """Test cases for DataStreamManager"""
    
    @pytest.fixture
    def stream_manager(self):
        """Create DataStreamManager for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        manager = DataStreamManager(temp_file.name)
        yield manager
        
        manager.close()
        os.unlink(temp_file.name)
    
    def test_financial_stream(self, stream_manager):
        """Test financial data stream generation"""
        data_point = stream_manager.create_financial_stream('TEST_STOCK')
        
        assert 'features' in data_point
        assert 'label' in data_point
        assert len(data_point['features']) == 15
        assert data_point['data_source'] == 'financial_TEST_STOCK'
    
    def test_iot_stream(self, stream_manager):
        """Test IoT sensor stream generation"""
        data_point = stream_manager.create_iot_sensor_stream('SENSOR_TEST')
        
        assert 'features' in data_point
        assert 'label' in data_point
        assert len(data_point['features']) == 15
        assert data_point['data_source'] == 'iot_SENSOR_TEST'
    
    def test_initial_data_population(self, stream_manager):
        """Test initial data population"""
        stream_manager.populate_initial_data(100)
        
        stats = stream_manager.db.get_data_statistics()
        assert stats['total_records'] == 100
    
    def test_mixed_batch_retrieval(self, stream_manager):
        """Test mixed batch data retrieval"""
        stream_manager.populate_initial_data(50)
        
        X, y = stream_manager.get_mixed_batch(30)
        
        assert len(X) <= 30  # May be less if not enough data
        assert len(y) == len(X)

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Create temporary database
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        try:
            # Initialize components
            ols = OnlineLearningSystem(random_state=42)
            stream_manager = DataStreamManager(temp_file.name)
            
            # Populate initial data
            stream_manager.populate_initial_data(200)
            
            # Get training data
            X, y = stream_manager.get_mixed_batch(200)
            
            # Train models
            results, X_test, y_test = ols.train_streaming(X, y, batch_size=50)
            
            # Verify results
            assert len(results) > 0
            assert all(isinstance(perf, ModelPerformance) for perf in results.values())
            
            # Store performance metrics
            for model_name, performance in results.items():
                stream_manager.db.store_performance_metrics(
                    model_name, 1, performance.accuracy, 
                    performance.precision, performance.recall, performance.f1_score
                )
            
            # Verify storage
            perf_history = stream_manager.db.get_performance_history()
            assert len(perf_history) == len(results)
            
        finally:
            # Cleanup
            stream_manager.close()
            os.unlink(temp_file.name)

# Performance benchmarks
class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_training_speed(self):
        """Test training speed benchmarks"""
        ols = OnlineLearningSystem(random_state=42)
        X, y = ols.generate_streaming_data(n_samples=1000, n_features=20)
        
        import time
        start_time = time.time()
        
        results, _, _ = ols.train_streaming(X, y, batch_size=100)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert training_time < 30  # 30 seconds threshold
        
        # All models should have reasonable performance
        for model_name, performance in results.items():
            assert performance.accuracy > 0.3  # Basic sanity check
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        ols = OnlineLearningSystem(random_state=42)
        
        # Train multiple times to check for memory leaks
        for _ in range(3):
            X, y = ols.generate_streaming_data(n_samples=500, n_features=10)
            ols.train_streaming(X, y, batch_size=50)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
