"""
Mock Database for Online Learning System
Simulates real-time data streams and storage for online learning algorithms
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MockStreamingDatabase:
    """
    Mock database that simulates real-time streaming data for online learning
    """
    
    def __init__(self, db_path: str = "streaming_data.db"):
        self.db_path = db_path
        self.connection = None
        self.streaming_active = False
        self.setup_database()
        
    def setup_database(self):
        """Initialize database tables"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.connection.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS streaming_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                features TEXT,
                label INTEGER,
                batch_id INTEGER,
                data_source TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                prediction INTEGER,
                confidence REAL,
                actual_label INTEGER,
                features TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                batch_id INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL
            )
        ''')
        
        self.connection.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def generate_synthetic_stream(self, n_features: int = 15, stream_rate: float = 1.0,
                                 concept_drift_probability: float = 0.01) -> Dict:
        """Generate a single synthetic data point for streaming"""
        
        # Base feature generation
        features = np.random.randn(n_features)
        
        # Add some structure to make classification meaningful
        if np.random.random() < concept_drift_probability:
            # Introduce concept drift
            features += np.random.normal(0, 2, n_features)
        
        # Generate label based on features with some noise
        linear_combination = np.sum(features[:5]) + np.random.normal(0, 0.5)
        label = 1 if linear_combination > 0 else 0
        
        return {
            'features': features.tolist(),
            'label': label,
            'timestamp': datetime.now(),
            'data_source': 'synthetic_stream'
        }
    
    def insert_streaming_data(self, data_point: Dict, batch_id: int = None):
        """Insert a single data point into the streaming table"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO streaming_data (features, label, batch_id, data_source)
            VALUES (?, ?, ?, ?)
        ''', (
            json.dumps(data_point['features']),
            data_point['label'],
            batch_id,
            data_point['data_source']
        ))
        
        self.connection.commit()
    
    def get_batch_data(self, batch_size: int = 100, batch_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve a batch of data from the database"""
        cursor = self.connection.cursor()
        
        if batch_id is not None:
            query = '''
                SELECT features, label FROM streaming_data 
                WHERE batch_id = ? 
                ORDER BY timestamp
            '''
            cursor.execute(query, (batch_id,))
        else:
            query = '''
                SELECT features, label FROM streaming_data 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            cursor.execute(query, (batch_size,))
        
        rows = cursor.fetchall()
        
        if not rows:
            return np.array([]), np.array([])
        
        features = np.array([json.loads(row[0]) for row in rows])
        labels = np.array([row[1] for row in rows])
        
        return features, labels
    
    def store_prediction(self, model_name: str, prediction: int, confidence: float,
                        actual_label: int, features: List[float]):
        """Store model prediction for evaluation"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO model_predictions (model_name, prediction, confidence, actual_label, features)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            model_name,
            prediction,
            confidence,
            actual_label,
            json.dumps(features)
        ))
        
        self.connection.commit()
    
    def store_performance_metrics(self, model_name: str, batch_id: int, 
                                 accuracy: float, precision: float, 
                                 recall: float, f1_score: float):
        """Store model performance metrics"""
        cursor = self.connection.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance (model_name, batch_id, accuracy, precision_score, recall, f1_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_name, batch_id, accuracy, precision, recall, f1_score
        ))
        
        self.connection.commit()
    
    def get_performance_history(self, model_name: str = None) -> pd.DataFrame:
        """Retrieve performance history for analysis"""
        cursor = self.connection.cursor()
        
        if model_name:
            query = '''
                SELECT * FROM model_performance 
                WHERE model_name = ? 
                ORDER BY timestamp
            '''
            cursor.execute(query, (model_name,))
        else:
            query = '''
                SELECT * FROM model_performance 
                ORDER BY timestamp
            '''
            cursor.execute(query)
        
        columns = ['id', 'timestamp', 'model_name', 'batch_id', 'accuracy', 
                  'precision_score', 'recall', 'f1_score']
        
        return pd.DataFrame(cursor.fetchall(), columns=columns)
    
    def simulate_real_time_stream(self, duration_minutes: int = 5, 
                                 points_per_minute: int = 60):
        """Simulate real-time data streaming"""
        logger.info(f"Starting real-time simulation for {duration_minutes} minutes")
        
        self.streaming_active = True
        start_time = datetime.now()
        batch_id = int(time.time())  # Use timestamp as batch ID
        
        total_points = duration_minutes * points_per_minute
        interval = 60.0 / points_per_minute  # seconds between points
        
        for i in range(total_points):
            if not self.streaming_active:
                break
                
            # Generate and insert data point
            data_point = self.generate_synthetic_stream()
            self.insert_streaming_data(data_point, batch_id)
            
            # Log progress
            if (i + 1) % points_per_minute == 0:
                elapsed = datetime.now() - start_time
                logger.info(f"Streamed {i + 1} points in {elapsed}")
            
            time.sleep(interval)
        
        logger.info("Real-time simulation completed")
    
    def start_background_streaming(self, duration_minutes: int = 10):
        """Start streaming in background thread"""
        streaming_thread = threading.Thread(
            target=self.simulate_real_time_stream,
            args=(duration_minutes,)
        )
        streaming_thread.daemon = True
        streaming_thread.start()
        return streaming_thread
    
    def stop_streaming(self):
        """Stop the streaming simulation"""
        self.streaming_active = False
        logger.info("Streaming stopped")
    
    def get_data_statistics(self) -> Dict:
        """Get statistics about stored data"""
        cursor = self.connection.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM streaming_data")
        total_records = cursor.fetchone()[0]
        
        # Label distribution
        cursor.execute("SELECT label, COUNT(*) FROM streaming_data GROUP BY label")
        label_dist = dict(cursor.fetchall())
        
        # Data sources
        cursor.execute("SELECT data_source, COUNT(*) FROM streaming_data GROUP BY data_source")
        source_dist = dict(cursor.fetchall())
        
        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM streaming_data")
        time_range = cursor.fetchone()
        
        return {
            'total_records': total_records,
            'label_distribution': label_dist,
            'source_distribution': source_dist,
            'time_range': time_range
        }
    
    def export_data(self, output_file: str = "exported_streaming_data.csv"):
        """Export streaming data to CSV"""
        query = '''
            SELECT timestamp, features, label, batch_id, data_source 
            FROM streaming_data 
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, self.connection)
        df.to_csv(output_file, index=False)
        logger.info(f"Data exported to {output_file}")
    
    def clear_data(self, table_name: str = None):
        """Clear data from specified table or all tables"""
        cursor = self.connection.cursor()
        
        if table_name:
            cursor.execute(f"DELETE FROM {table_name}")
        else:
            cursor.execute("DELETE FROM streaming_data")
            cursor.execute("DELETE FROM model_predictions")
            cursor.execute("DELETE FROM model_performance")
        
        self.connection.commit()
        logger.info(f"Cleared data from {table_name or 'all tables'}")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class DataStreamManager:
    """
    Manager class for handling multiple data streams and sources
    """
    
    def __init__(self, db_path: str = "streaming_data.db"):
        self.db = MockStreamingDatabase(db_path)
        self.active_streams = {}
    
    def create_financial_stream(self, symbol: str = "MOCK_STOCK") -> Dict:
        """Simulate financial market data stream"""
        base_price = 100.0
        volatility = 0.02
        
        # Generate price movement
        price_change = np.random.normal(0, volatility)
        current_price = base_price * (1 + price_change)
        
        # Create features: price, volume, moving averages, etc.
        features = [
            current_price,
            np.random.randint(1000, 10000),  # volume
            current_price + np.random.normal(0, 0.5),  # moving avg
            np.random.uniform(-1, 1),  # momentum
            np.random.uniform(0, 100),  # RSI-like indicator
        ]
        
        # Extend to desired feature count
        features.extend(np.random.randn(10))
        
        # Label: 1 if price will go up, 0 if down (simplified)
        label = 1 if price_change > 0 else 0
        
        return {
            'features': features,
            'label': label,
            'timestamp': datetime.now(),
            'data_source': f'financial_{symbol}'
        }
    
    def create_iot_sensor_stream(self, sensor_id: str = "SENSOR_001") -> Dict:
        """Simulate IoT sensor data stream"""
        # Simulate sensor readings
        temperature = np.random.normal(25, 5)  # Celsius
        humidity = np.random.uniform(30, 80)   # Percentage
        pressure = np.random.normal(1013, 10)  # hPa
        
        features = [
            temperature,
            humidity,
            pressure,
            np.random.uniform(0, 100),  # light level
            np.random.uniform(0, 50),   # noise level
        ]
        
        # Add more synthetic features
        features.extend(np.random.randn(10))
        
        # Anomaly detection: label 1 if anomalous conditions
        anomaly = (temperature > 35 or temperature < 5 or 
                  humidity > 90 or pressure < 980)
        label = 1 if anomaly else 0
        
        return {
            'features': features,
            'label': label,
            'timestamp': datetime.now(),
            'data_source': f'iot_{sensor_id}'
        }
    
    def populate_initial_data(self, n_samples: int = 1000):
        """Populate database with initial training data"""
        logger.info(f"Populating database with {n_samples} initial samples")
        
        for i in range(n_samples):
            # Mix different data sources
            if i % 3 == 0:
                data_point = self.create_financial_stream()
            elif i % 3 == 1:
                data_point = self.create_iot_sensor_stream()
            else:
                data_point = self.db.generate_synthetic_stream()
            
            self.db.insert_streaming_data(data_point, batch_id=0)
        
        logger.info("Initial data population completed")
    
    def get_mixed_batch(self, batch_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get a mixed batch from different data sources"""
        return self.db.get_batch_data(batch_size)
    
    def close(self):
        """Close all connections and stop streams"""
        self.db.stop_streaming()
        self.db.close()

def main():
    """Demonstration of mock database functionality"""
    logger.info("Initializing Mock Streaming Database...")
    
    # Create database manager
    stream_manager = DataStreamManager()
    
    # Populate with initial data
    stream_manager.populate_initial_data(500)
    
    # Get statistics
    stats = stream_manager.db.get_data_statistics()
    logger.info(f"Database statistics: {stats}")
    
    # Start background streaming
    logger.info("Starting background data stream...")
    stream_thread = stream_manager.db.start_background_streaming(duration_minutes=2)
    
    # Simulate some batch processing
    for i in range(5):
        time.sleep(10)  # Wait 10 seconds
        X_batch, y_batch = stream_manager.get_mixed_batch(50)
        logger.info(f"Retrieved batch {i+1}: {X_batch.shape[0]} samples")
    
    # Stop streaming and close
    stream_manager.close()
    logger.info("Mock database demonstration completed")

if __name__ == "__main__":
    main()
