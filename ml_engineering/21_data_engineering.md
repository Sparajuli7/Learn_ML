# Data Engineering for Machine Learning: Building the Foundation
*"Data is the new oil, but unlike oil, data becomes more valuable when refined and connected"*

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [Data Engineering Fundamentals](#data-engineering-fundamentals)
3. [Data Pipeline Architecture](#data-pipeline-architecture)
4. [Data Quality and Validation](#data-quality-and-validation)
5. [Real-time Data Processing](#real-time-data-processing)
6. [Data Storage and Retrieval](#data-storage-and-retrieval)
7. [Implementation Examples](#implementation-examples)
8. [Exercises and Projects](#exercises-and-projects)
9. [Further Reading](#further-reading)

## 🎯 Introduction

Data engineering is the backbone of any successful machine learning system. Think of it as the plumbing system of a house - you might not see it, but without proper pipes, water (or in our case, data) can't flow where it needs to go. In 2025, with the explosion of AI applications, data engineering has become more critical than ever.

### Why Data Engineering Matters in 2025

The AI landscape in 2025 is characterized by:
- **Massive Scale**: Companies processing petabytes of data daily
- **Real-time Requirements**: Instant decision-making needs
- **Multi-modal Data**: Text, images, audio, video, and structured data
- **Regulatory Compliance**: GDPR, CCPA, and emerging AI regulations
- **Cost Optimization**: Efficient data processing to manage cloud costs

### Historical Context

Data engineering evolved from simple ETL (Extract, Transform, Load) processes in the 1990s to complex real-time streaming systems today. The journey includes:

- **1990s**: Batch processing with mainframes
- **2000s**: Data warehouses and OLAP
- **2010s**: Big Data with Hadoop and Spark
- **2020s**: Real-time streaming and cloud-native
- **2025**: AI-native data engineering with automated pipelines

## 🧮 Mathematical Foundations

### Data Flow Optimization

The efficiency of a data pipeline can be measured using several mathematical concepts:

#### 1. Throughput Rate (T)
```
T = (Number of records processed) / (Time taken)
```

#### 2. Latency (L)
```
L = Processing time + Network time + Storage time
```

#### 3. Data Quality Score (DQS)
```
DQS = (Valid records) / (Total records) × 100
```

#### 4. Cost Efficiency (CE)
```
CE = (Processing power) / (Cost per hour)
```

### Example Calculation

Let's say we're processing 1 million records:
- Processing time: 2 hours
- Valid records: 950,000
- Cost: $50/hour

```
T = 1,000,000 / 2 = 500,000 records/hour
DQS = (950,000 / 1,000,000) × 100 = 95%
CE = 500,000 / 50 = 10,000 records/hour/dollar
```

## 💻 Implementation

### 1. Building a Data Pipeline with Apache Airflow

Apache Airflow is like a smart traffic controller for data - it ensures all your data flows arrive at the right place at the right time.

```python
# Why: Airflow provides reliable, scalable workflow management
# How: DAGs (Directed Acyclic Graphs) define task dependencies
# Where: Production environments requiring complex data workflows
# What: Automated data pipeline execution with monitoring
# When: When you need reliable, scheduled data processing

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging

# Define default arguments for our DAG
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

# Create our DAG
dag = DAG(
    'ml_data_pipeline',
    default_args=default_args,
    description='ML Data Processing Pipeline',
    schedule_interval=timedelta(hours=1),
    catchup=False
)

def extract_data():
    """Extract data from various sources"""
    # Why: Centralized data extraction from multiple sources
    # How: Connect to databases, APIs, and file systems
    # Where: First step in any data pipeline
    # What: Raw data collection
    # When: Scheduled or triggered by events
    
    logging.info("Starting data extraction...")
    
    # Simulate extracting from multiple sources
    sources = {
        'user_behavior': 'https://api.example.com/user-data',
        'product_catalog': 'https://api.example.com/products',
        'transaction_history': 'https://api.example.com/transactions'
    }
    
    extracted_data = {}
    for source_name, url in sources.items():
        try:
            # In real implementation, you'd make actual API calls
            # For demonstration, we'll create sample data
            if source_name == 'user_behavior':
                data = pd.DataFrame({
                    'user_id': range(1, 1001),
                    'page_views': np.random.randint(1, 50, 1000),
                    'session_duration': np.random.randint(30, 1800, 1000),
                    'timestamp': pd.date_range('2025-01-01', periods=1000, freq='H')
                })
            elif source_name == 'product_catalog':
                data = pd.DataFrame({
                    'product_id': range(1, 501),
                    'category': np.random.choice(['electronics', 'clothing', 'books'], 500),
                    'price': np.random.uniform(10, 1000, 500),
                    'rating': np.random.uniform(1, 5, 500)
                })
            else:  # transaction_history
                data = pd.DataFrame({
                    'transaction_id': range(1, 2001),
                    'user_id': np.random.randint(1, 1001, 2000),
                    'amount': np.random.uniform(5, 500, 2000),
                    'timestamp': pd.date_range('2025-01-01', periods=2000, freq='30min')
                })
            
            extracted_data[source_name] = data
            logging.info(f"Successfully extracted {len(data)} records from {source_name}")
            
        except Exception as e:
            logging.error(f"Failed to extract from {source_name}: {str(e)}")
            raise
    
    return extracted_data

def transform_data(**context):
    """Transform and clean the extracted data"""
    # Why: Raw data needs cleaning and transformation for ML models
    # How: Apply business rules, handle missing values, create features
    # Where: After extraction, before loading
    # What: Clean, structured data ready for ML
    # When: Always needed before model training
    
    logging.info("Starting data transformation...")
    
    # Get data from previous task
    ti = context['task_instance']
    extracted_data = ti.xcom_pull(task_ids='extract_data')
    
    transformed_data = {}
    
    for source_name, data in extracted_data.items():
        logging.info(f"Transforming {source_name} data...")
        
        # Handle missing values
        data = data.fillna(method='ffill')
        
        # Add derived features
        if source_name == 'user_behavior':
            data['engagement_score'] = (
                data['page_views'] * 0.3 + 
                data['session_duration'] * 0.7
            )
            data['is_active_user'] = data['engagement_score'] > data['engagement_score'].median()
            
        elif source_name == 'product_catalog':
            data['price_category'] = pd.cut(
                data['price'], 
                bins=[0, 50, 200, 1000], 
                labels=['budget', 'mid-range', 'premium']
            )
            
        elif source_name == 'transaction_history':
            data['hour_of_day'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].isin([5, 6])
        
        transformed_data[source_name] = data
        logging.info(f"Transformed {source_name}: {len(data)} records")
    
    return transformed_data

def load_data(**context):
    """Load transformed data to data warehouse"""
    # Why: Centralized storage for ML model access
    # How: Write to database or data lake
    # Where: Final step in data pipeline
    # What: Persistent storage of processed data
    # When: After transformation is complete
    
    logging.info("Starting data loading...")
    
    ti = context['task_instance']
    transformed_data = ti.xcom_pull(task_ids='transform_data')
    
    # In real implementation, you'd write to actual database
    # For demonstration, we'll save to files
    for source_name, data in transformed_data.items():
        filename = f"/tmp/processed_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
        data.to_parquet(filename, index=False)
        logging.info(f"Saved {source_name} to {filename}")
    
    logging.info("Data loading completed successfully")

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

# Set task dependencies
extract_task >> transform_task >> load_task
```

### 2. Real-time Data Processing with Apache Kafka

Kafka is like a high-speed conveyor belt for data - it moves information quickly and reliably between different parts of your system.

```python
# Why: Real-time data processing for immediate ML predictions
# How: Stream processing with fault tolerance
# Where: Systems requiring low-latency data processing
# What: Continuous data flow with message queuing
# When: When you need real-time analytics or predictions

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class RealTimeDataProcessor:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        # Why: Initialize connection to Kafka cluster
        # How: Connect to Kafka brokers
        # Where: Any real-time data processing system
        # What: Producer and consumer setup
        # When: At system startup
        
        self.bootstrap_servers = bootstrap_servers
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            'user_events',
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='ml_processor_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logging.info("Real-time data processor initialized")
    
    def produce_user_event(self, user_id, event_type, event_data):
        """Produce a user event to Kafka"""
        # Why: Send user events for real-time processing
        # How: Serialize and send to Kafka topic
        # Where: User-facing applications
        # What: Event data for ML model input
        # When: When user performs actions
        
        event = {
            'user_id': user_id,
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            future = self.producer.send('user_events', event)
            future.get(timeout=10)  # Wait for send confirmation
            logging.info(f"Produced event for user {user_id}: {event_type}")
        except KafkaError as e:
            logging.error(f"Failed to produce event: {e}")
    
    def process_user_events(self):
        """Process incoming user events in real-time"""
        # Why: Real-time feature engineering and prediction
        # How: Stream processing with sliding windows
        # Where: ML inference pipelines
        # What: Instant predictions and recommendations
        # When: Continuously as events arrive
        
        logging.info("Starting real-time event processing...")
        
        # Initialize user state tracking
        user_states = {}
        
        for message in self.consumer:
            event = message.value
            user_id = event['user_id']
            
            # Update user state
            if user_id not in user_states:
                user_states[user_id] = {
                    'event_count': 0,
                    'last_event_time': None,
                    'event_types': set(),
                    'total_value': 0
                }
            
            state = user_states[user_id]
            state['event_count'] += 1
            state['last_event_time'] = event['timestamp']
            state['event_types'].add(event['event_type'])
            
            # Add value if it's a purchase event
            if event['event_type'] == 'purchase':
                state['total_value'] += event['event_data'].get('amount', 0)
            
            # Generate real-time features
            features = self._extract_features(user_id, state, event)
            
            # Make prediction (simplified)
            prediction = self._make_prediction(features)
            
            # Send prediction back to user
            self._send_prediction(user_id, prediction)
            
            logging.info(f"Processed event for user {user_id}, prediction: {prediction}")
    
    def _extract_features(self, user_id, state, current_event):
        """Extract features for ML model"""
        # Why: Convert raw events into ML model features
        # How: Feature engineering from event data
        # Where: Real-time ML inference
        # What: Numerical features for prediction
        # When: For each incoming event
        
        features = {
            'user_id': user_id,
            'event_count': state['event_count'],
            'unique_event_types': len(state['event_types']),
            'total_value': state['total_value'],
            'avg_value_per_event': state['total_value'] / max(state['event_count'], 1),
            'is_purchase_event': 1 if current_event['event_type'] == 'purchase' else 0,
            'hour_of_day': datetime.fromisoformat(current_event['timestamp']).hour,
            'day_of_week': datetime.fromisoformat(current_event['timestamp']).weekday()
        }
        
        return features
    
    def _make_prediction(self, features):
        """Make prediction using ML model"""
        # Why: Provide real-time predictions to users
        # How: Apply trained ML model to features
        # Where: Real-time recommendation systems
        # What: Personalized predictions and recommendations
        # When: For each user interaction
        
        # Simplified prediction logic
        # In real implementation, you'd load a trained model
        score = (
            features['event_count'] * 0.1 +
            features['unique_event_types'] * 0.2 +
            features['total_value'] * 0.001 +
            features['is_purchase_event'] * 0.5
        )
        
        if score > 0.7:
            return "high_value_user"
        elif score > 0.3:
            return "medium_value_user"
        else:
            return "low_value_user"
    
    def _send_prediction(self, user_id, prediction):
        """Send prediction back to user"""
        # Why: Provide immediate feedback to users
        # How: Send to user interface or notification system
        # Where: User-facing applications
        # What: Real-time user experience improvements
        # When: Immediately after prediction
        
        response = {
            'user_id': user_id,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat(),
            'confidence': 0.85  # Simplified confidence score
        }
        
        # In real implementation, send to user interface
        logging.info(f"Sent prediction to user {user_id}: {prediction}")

# Usage example
if __name__ == "__main__":
    processor = RealTimeDataProcessor()
    
    # Simulate producing events
    processor.produce_user_event(123, 'page_view', {'page': '/products'})
    processor.produce_user_event(123, 'purchase', {'amount': 99.99})
    
    # Start processing (in real implementation, this would run continuously)
    processor.process_user_events()
```

### 3. Data Quality Monitoring with Great Expectations

Great Expectations is like a quality control inspector for your data - it ensures your data meets the standards your ML models expect.

```python
# Why: Ensure data quality for reliable ML models
# How: Automated data validation and monitoring
# Where: Data pipelines and ML workflows
# What: Data quality checks and alerts
# When: Before model training and in production

import great_expectations as ge
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class DataQualityMonitor:
    def __init__(self):
        # Why: Initialize data quality monitoring system
        # How: Set up Great Expectations context
        # Where: Data engineering pipelines
        # What: Automated data validation framework
        # When: At system startup
        
        self.context = ge.get_context()
        logging.info("Data quality monitor initialized")
    
    def validate_user_data(self, data):
        """Validate user behavior data"""
        # Why: Ensure user data meets quality standards
        # How: Apply statistical and business rule checks
        # Where: User data processing pipelines
        # What: Quality validation results
        # When: Before ML model training
        
        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(data)
        
        # Define expectations (data quality rules)
        expectations = [
            # Check for required columns
            ge_df.expect_table_columns_to_match_ordered_list([
                'user_id', 'page_views', 'session_duration', 'timestamp'
            ]),
            
            # Check data types
            ge_df.expect_column_values_to_be_of_type('user_id', 'int64'),
            ge_df.expect_column_values_to_be_of_type('page_views', 'int64'),
            ge_df.expect_column_values_to_be_of_type('session_duration', 'int64'),
            
            # Check for missing values
            ge_df.expect_column_values_to_not_be_null('user_id'),
            ge_df.expect_column_values_to_not_be_null('timestamp'),
            
            # Check value ranges
            ge_df.expect_column_values_to_be_between('page_views', 0, 1000),
            ge_df.expect_column_values_to_be_between('session_duration', 0, 7200),  # 2 hours max
            
            # Check for duplicates
            ge_df.expect_compound_columns_to_be_unique(['user_id', 'timestamp']),
            
            # Check timestamp format
            ge_df.expect_column_values_to_match_regex('timestamp', r'\d{4}-\d{2}-\d{2}')
        ]
        
        # Run validations
        validation_results = []
        for expectation in expectations:
            result = expectation.run()
            validation_results.append(result)
            
            if not result.success:
                logging.warning(f"Data quality check failed: {result.expectation_config.expectation_type}")
        
        # Generate quality report
        quality_score = self._calculate_quality_score(validation_results)
        
        return {
            'passed': all(r.success for r in validation_results),
            'quality_score': quality_score,
            'results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_product_data(self, data):
        """Validate product catalog data"""
        # Why: Ensure product data is consistent and complete
        # How: Apply product-specific validation rules
        # Where: Product data processing pipelines
        # What: Product data quality validation
        # When: Before recommendation model training
        
        ge_df = ge.from_pandas(data)
        
        expectations = [
            # Check required columns
            ge_df.expect_table_columns_to_match_ordered_list([
                'product_id', 'category', 'price', 'rating'
            ]),
            
            # Check data types
            ge_df.expect_column_values_to_be_of_type('product_id', 'int64'),
            ge_df.expect_column_values_to_be_of_type('price', 'float64'),
            ge_df.expect_column_values_to_be_of_type('rating', 'float64'),
            
            # Check value ranges
            ge_df.expect_column_values_to_be_between('price', 0, 10000),
            ge_df.expect_column_values_to_be_between('rating', 0, 5),
            
            # Check for valid categories
            ge_df.expect_column_values_to_be_in_set('category', 
                ['electronics', 'clothing', 'books', 'home', 'sports']),
            
            # Check for unique product IDs
            ge_df.expect_column_values_to_be_unique('product_id')
        ]
        
        validation_results = []
        for expectation in expectations:
            result = expectation.run()
            validation_results.append(result)
        
        quality_score = self._calculate_quality_score(validation_results)
        
        return {
            'passed': all(r.success for r in validation_results),
            'quality_score': quality_score,
            'results': validation_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_quality_score(self, validation_results):
        """Calculate overall data quality score"""
        # Why: Provide quantitative measure of data quality
        # How: Weighted average of validation results
        # Where: Data quality monitoring systems
        # What: Percentage score of data quality
        # When: After running all validations
        
        if not validation_results:
            return 0.0
        
        passed_checks = sum(1 for r in validation_results if r.success)
        total_checks = len(validation_results)
        
        return (passed_checks / total_checks) * 100
    
    def generate_quality_report(self, validation_results):
        """Generate comprehensive quality report"""
        # Why: Provide detailed insights into data quality
        # How: Aggregate validation results into report
        # Where: Data engineering dashboards
        # What: Detailed quality analysis
        # When: After validation runs
        
        report = {
            'summary': {
                'total_checks': len(validation_results),
                'passed_checks': sum(1 for r in validation_results if r.success),
                'failed_checks': sum(1 for r in validation_results if not r.success),
                'quality_score': self._calculate_quality_score(validation_results)
            },
            'details': []
        }
        
        for result in validation_results:
            detail = {
                'expectation_type': result.expectation_config.expectation_type,
                'success': result.success,
                'observed_value': result.result.get('observed_value'),
                'expected_value': result.expectation_config.kwargs
            }
            report['details'].append(detail)
        
        return report

# Usage example
if __name__ == "__main__":
    monitor = DataQualityMonitor()
    
    # Sample data
    user_data = pd.DataFrame({
        'user_id': range(1, 101),
        'page_views': np.random.randint(1, 50, 100),
        'session_duration': np.random.randint(30, 1800, 100),
        'timestamp': pd.date_range('2025-01-01', periods=100, freq='H')
    })
    
    # Validate data
    validation_result = monitor.validate_user_data(user_data)
    print(f"Data quality score: {validation_result['quality_score']:.1f}%")
    
    if validation_result['passed']:
        print("✅ Data quality checks passed")
    else:
        print("❌ Data quality checks failed")
```

## 🎯 Applications

### 1. E-commerce Recommendation System

**Problem**: An e-commerce platform needs to provide real-time product recommendations based on user behavior.

**Solution**: 
- **Data Pipeline**: Kafka streams user clicks, purchases, and browsing data
- **Feature Engineering**: Real-time calculation of user preferences, product popularity
- **ML Model**: Collaborative filtering with real-time updates
- **Results**: 25% increase in conversion rate, 40% reduction in latency

### 2. Healthcare Data Processing

**Problem**: A hospital needs to process patient data for predictive diagnostics while maintaining privacy.

**Solution**:
- **Data Pipeline**: HIPAA-compliant data processing with encryption
- **Quality Monitoring**: Automated validation of medical data accuracy
- **Real-time Processing**: Immediate alert generation for critical conditions
- **Results**: 30% faster diagnosis, 99.9% data accuracy

### 3. Financial Fraud Detection

**Problem**: A bank needs to detect fraudulent transactions in real-time.

**Solution**:
- **Data Pipeline**: Real-time transaction streaming with sub-second latency
- **Feature Engineering**: Risk scoring based on transaction patterns
- **Quality Assurance**: Continuous monitoring of model performance
- **Results**: 95% fraud detection rate, 0.1% false positives

## 🧪 Exercises and Projects

### Exercise 1: Build a Simple Data Pipeline

Create a data pipeline that processes daily sales data:

```python
# Your task: Build a complete ETL pipeline
# Requirements:
# 1. Extract data from CSV files
# 2. Transform: Calculate daily totals, add date features
# 3. Load: Save to database
# 4. Add data quality checks
# 5. Implement error handling

# Starter code:
import pandas as pd
import sqlite3
from datetime import datetime

def extract_sales_data(file_path):
    """Extract sales data from CSV file"""
    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        
        # Basic data validation
        required_columns = ['date', 'product_id', 'quantity', 'price', 'customer_id']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert date column to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        print(f"Successfully extracted {len(data)} records from {file_path}")
        return data
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

def transform_sales_data(data):
    """Transform sales data"""
    if data is None:
        return None
    
    try:
        # Create a copy to avoid modifying original data
        transformed_data = data.copy()
        
        # Calculate total sales amount
        transformed_data['total_amount'] = transformed_data['quantity'] * transformed_data['price']
        
        # Extract date features
        transformed_data['year'] = transformed_data['date'].dt.year
        transformed_data['month'] = transformed_data['date'].dt.month
        transformed_data['day'] = transformed_data['date'].dt.day
        transformed_data['day_of_week'] = transformed_data['date'].dt.dayofweek
        
        # Create customer features
        customer_stats = transformed_data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'product_id': 'nunique'
        }).round(2)
        
        customer_stats.columns = ['total_spent', 'avg_order_value', 'order_count', 'unique_products']
        customer_stats = customer_stats.reset_index()
        
        # Merge customer features back to main data
        transformed_data = transformed_data.merge(customer_stats, on='customer_id', how='left')
        
        # Create product features
        product_stats = transformed_data.groupby('product_id').agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'customer_id': 'nunique'
        }).round(2)
        
        product_stats.columns = ['total_quantity_sold', 'total_revenue', 'unique_customers']
        product_stats = product_stats.reset_index()
        
        # Merge product features back to main data
        transformed_data = transformed_data.merge(product_stats, on='product_id', how='left')
        
        print(f"Transformed data shape: {transformed_data.shape}")
        return transformed_data
        
    except Exception as e:
        print(f"Error transforming data: {e}")
        return None

def load_sales_data(data, db_path):
    """Load data to database"""
    if data is None:
        return False
    
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        
        # Load main sales data
        data.to_sql('sales_data', conn, if_exists='replace', index=False)
        
        # Create summary tables
        daily_summary = data.groupby('date').agg({
            'total_amount': 'sum',
            'quantity': 'sum',
            'customer_id': 'nunique',
            'product_id': 'nunique'
        }).reset_index()
        daily_summary.columns = ['date', 'daily_revenue', 'daily_quantity', 'daily_customers', 'daily_products']
        
        daily_summary.to_sql('daily_summary', conn, if_exists='replace', index=False)
        
        # Create customer summary
        customer_summary = data.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'product_id': 'nunique',
            'date': lambda x: (x.max() - x.min()).days
        }).round(2)
        customer_summary.columns = ['total_spent', 'avg_order_value', 'order_count', 'unique_products', 'customer_lifetime_days']
        customer_summary = customer_summary.reset_index()
        
        customer_summary.to_sql('customer_summary', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Successfully loaded data to {db_path}")
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def validate_sales_data(data):
    """Validate sales data quality"""
    if data is None:
        return False
    
    validation_results = {
        'total_records': len(data),
        'missing_values': data.isnull().sum().sum(),
        'duplicate_records': data.duplicated().sum(),
        'negative_prices': (data['price'] < 0).sum(),
        'negative_quantities': (data['quantity'] < 0).sum(),
        'zero_prices': (data['price'] == 0).sum(),
        'zero_quantities': (data['quantity'] == 0).sum(),
        'date_range': {
            'min_date': data['date'].min(),
            'max_date': data['date'].max()
        }
    }
    
    # Check for data quality issues
    issues = []
    
    if validation_results['missing_values'] > 0:
        issues.append(f"Found {validation_results['missing_values']} missing values")
    
    if validation_results['duplicate_records'] > 0:
        issues.append(f"Found {validation_results['duplicate_records']} duplicate records")
    
    if validation_results['negative_prices'] > 0:
        issues.append(f"Found {validation_results['negative_prices']} negative prices")
    
    if validation_results['negative_quantities'] > 0:
        issues.append(f"Found {validation_results['negative_quantities']} negative quantities")
    
    if validation_results['zero_prices'] > 0:
        issues.append(f"Found {validation_results['zero_prices']} zero prices")
    
    if validation_results['zero_quantities'] > 0:
        issues.append(f"Found {validation_results['zero_quantities']} zero quantities")
    
    # Print validation results
    print("Data Validation Results:")
    print(f"Total records: {validation_results['total_records']}")
    print(f"Date range: {validation_results['date_range']['min_date']} to {validation_results['date_range']['max_date']}")
    
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Data validation passed!")
        return True
```

### Exercise 2: Real-time Data Processing

Build a real-time system that processes user events:

```python
# Your task: Create a real-time event processor
# Requirements:
# 1. Accept user events (clicks, purchases, etc.)
# 2. Calculate real-time features
# 3. Make predictions
# 4. Store results
# 5. Handle errors gracefully

# Starter code:
class RealTimeProcessor:
    def __init__(self):
        # Initialize processor with storage and models
        self.event_buffer = []
        self.user_profiles = {}
        self.model = None  # Placeholder for ML model
        self.feature_store = {}
        self.prediction_history = []
        
        # Configuration
        self.buffer_size = 1000
        self.processing_interval = 60  # seconds
        self.last_processing_time = datetime.now()
    
    def process_event(self, event):
        """Process a single user event"""
        try:
            # Validate event structure
            required_fields = ['user_id', 'event_type', 'timestamp', 'data']
            if not all(field in event for field in required_fields):
                raise ValueError(f"Missing required fields in event: {required_fields}")
            
            # Add event to buffer
            self.event_buffer.append(event)
            
            # Update user profile
            user_id = event['user_id']
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = {
                    'events': [],
                    'last_activity': event['timestamp'],
                    'total_events': 0,
                    'event_types': {}
                }
            
            # Update user statistics
            self.user_profiles[user_id]['events'].append(event)
            self.user_profiles[user_id]['last_activity'] = event['timestamp']
            self.user_profiles[user_id]['total_events'] += 1
            
            # Update event type counts
            event_type = event['event_type']
            if event_type not in self.user_profiles[user_id]['event_types']:
                self.user_profiles[user_id]['event_types'][event_type] = 0
            self.user_profiles[user_id]['event_types'][event_type] += 1
            
            # Process buffer if it's full or enough time has passed
            if len(self.event_buffer) >= self.buffer_size or \
               (datetime.now() - self.last_processing_time).seconds >= self.processing_interval:
                self._process_buffer()
            
            return True
            
        except Exception as e:
            print(f"Error processing event: {e}")
            return False
    
    def calculate_features(self, user_events):
        """Calculate features from user events"""
        if not user_events:
            return {}
        
        try:
            # Basic user features
            features = {
                'total_events': len(user_events),
                'unique_event_types': len(set(event['event_type'] for event in user_events)),
                'avg_events_per_day': len(user_events) / max(1, (user_events[-1]['timestamp'] - user_events[0]['timestamp']).days),
                'last_activity_days': (datetime.now() - user_events[-1]['timestamp']).days
            }
            
            # Event type distribution
            event_type_counts = {}
            for event in user_events:
                event_type = event['event_type']
                event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            
            # Add event type features
            for event_type, count in event_type_counts.items():
                features[f'{event_type}_count'] = count
                features[f'{event_type}_ratio'] = count / len(user_events)
            
            # Time-based features
            recent_events = [e for e in user_events if (datetime.now() - e['timestamp']).days <= 7]
            features['recent_activity_7d'] = len(recent_events)
            
            # Session features (assuming events within 30 minutes are in same session)
            sessions = []
            current_session = [user_events[0]]
            
            for i in range(1, len(user_events)):
                time_diff = (user_events[i]['timestamp'] - user_events[i-1]['timestamp']).total_seconds() / 60
                if time_diff <= 30:  # 30 minutes threshold
                    current_session.append(user_events[i])
                else:
                    sessions.append(current_session)
                    current_session = [user_events[i]]
            
            if current_session:
                sessions.append(current_session)
            
            features['total_sessions'] = len(sessions)
            features['avg_session_length'] = sum(len(session) for session in sessions) / len(sessions) if sessions else 0
            
            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return {}
    
    def make_prediction(self, features):
        """Make prediction based on features"""
        try:
            # Simple rule-based prediction (replace with actual ML model)
            prediction_score = 0
            
            # Scoring based on features
            if features.get('total_events', 0) > 10:
                prediction_score += 0.3
            
            if features.get('recent_activity_7d', 0) > 5:
                prediction_score += 0.3
            
            if features.get('avg_events_per_day', 0) > 2:
                prediction_score += 0.2
            
            if features.get('total_sessions', 0) > 3:
                prediction_score += 0.2
            
            # Determine prediction category
            if prediction_score >= 0.8:
                prediction = 'high_engagement'
            elif prediction_score >= 0.5:
                prediction = 'medium_engagement'
            else:
                prediction = 'low_engagement'
            
            # Store prediction
            prediction_record = {
                'timestamp': datetime.now(),
                'features': features,
                'prediction': prediction,
                'score': prediction_score
            }
            self.prediction_history.append(prediction_record)
            
            return {
                'prediction': prediction,
                'score': prediction_score,
                'confidence': min(prediction_score + 0.2, 1.0)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                'prediction': 'unknown',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def _process_buffer(self):
        """Process events in buffer"""
        if not self.event_buffer:
            return
        
        try:
            # Group events by user
            user_events = {}
            for event in self.event_buffer:
                user_id = event['user_id']
                if user_id not in user_events:
                    user_events[user_id] = []
                user_events[user_id].append(event)
            
            # Process each user's events
            for user_id, events in user_events.items():
                # Calculate features
                features = self.calculate_features(events)
                
                # Make prediction
                prediction = self.make_prediction(features)
                
                # Store results
                self.feature_store[user_id] = {
                    'features': features,
                    'prediction': prediction,
                    'last_updated': datetime.now()
                }
            
            # Clear buffer
            self.event_buffer = []
            self.last_processing_time = datetime.now()
            
            print(f"Processed {len(user_events)} users from buffer")
            
        except Exception as e:
            print(f"Error processing buffer: {e}")
```

### Project: Complete ML Data Pipeline

Build a comprehensive data pipeline for a recommendation system:

**Requirements**:
1. **Data Sources**: User behavior, product catalog, transactions
2. **Real-time Processing**: Kafka for streaming data
3. **Batch Processing**: Daily aggregations and model retraining
4. **Quality Monitoring**: Automated data validation
5. **Monitoring**: Performance metrics and alerting
6. **Documentation**: Complete system documentation

**Deliverables**:
- Working data pipeline code
- Architecture diagram
- Performance benchmarks
- Quality metrics report
- Deployment guide

## 📖 Further Reading

### Essential Resources

1. **Books**:
   - "Designing Data-Intensive Applications" by Martin Kleppmann
   - "The Data Warehouse Toolkit" by Ralph Kimball
   - "Streaming Systems" by Tyler Akidau

2. **Online Courses**:
   - DataCamp: "Data Engineering with Python"
   - Coursera: "Big Data Analysis with Scala and Spark"
   - edX: "Data Engineering Fundamentals"

3. **Tools and Technologies**:
   - **Apache Airflow**: Workflow management
   - **Apache Kafka**: Stream processing
   - **Apache Spark**: Big data processing
   - **Great Expectations**: Data quality
   - **dbt**: Data transformation
   - **Fivetran**: Data integration

4. **2025 Trends**:
   - **AI-Native Data Engineering**: Automated pipeline optimization
   - **Data Mesh**: Decentralized data architecture
   - **Real-time Everything**: Sub-second data processing
   - **Cost Optimization**: Efficient cloud resource usage
   - **Privacy-First**: Built-in data governance

### Certification Path

1. **Beginner**: Google Cloud Data Engineer
2. **Intermediate**: AWS Data Analytics Specialty
3. **Advanced**: Apache Kafka Developer
4. **Expert**: Data Engineering with Spark

## 🎯 Key Takeaways

1. **Data engineering is the foundation** of successful ML systems
2. **Real-time processing** is essential for modern AI applications
3. **Data quality** directly impacts model performance
4. **Automation** reduces errors and increases efficiency
5. **Monitoring** ensures system reliability
6. **Scalability** must be built in from the start

*"The best data pipeline is the one you don't have to think about"*

**Next: [ML Infrastructure](ml_engineering/22_ml_infrastructure.md) → Building scalable ML systems and deployment architectures**