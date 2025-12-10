"""
Integration tests for SageMaker deployment

These tests verify end-to-end workflows with AWS services.
Mark as @pytest.mark.integration to skip in CI without AWS credentials.
"""

import pytest
import boto3
import json
from unittest.mock import patch, MagicMock
from moto import mock_s3, mock_sagemaker
from src.ml_toolkit.sagemaker_utils import SageMakerDeployment
from src.ml_toolkit.config import AWSConfig


@pytest.mark.integration
class TestSageMakerDeployment:
    """Test SageMaker deployment workflows"""
    
    @mock_s3
    def test_upload_model_to_s3(self, temp_model_dir, sample_trained_model):
        """Test model artifact upload to S3"""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-ml-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        deployment = SageMakerDeployment(bucket_name=bucket_name)
        
        # Save model
        import joblib
        model_path = temp_model_dir / "model.pkl"
        joblib.dump(sample_trained_model, model_path)
        
        # Upload
        s3_uri = deployment.upload_model(str(model_path), prefix='models/test')
        
        assert s3_uri.startswith(f's3://{bucket_name}/models/test')
        
        # Verify file exists in S3
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='models/test')
        assert 'Contents' in objects
    
    @mock_sagemaker
    @mock_s3
    def test_create_model(self, mock_sagemaker_session):
        """Test SageMaker model creation"""
        config = AWSConfig()
        deployment = SageMakerDeployment()
        
        with patch('boto3.client') as mock_client:
            mock_sm = MagicMock()
            mock_client.return_value = mock_sm
            
            mock_sm.create_model.return_value = {
                'ModelArn': 'arn:aws:sagemaker:us-east-1:123456789:model/test-model'
            }
            
            model_name = 'test-fraud-model'
            model_data = 's3://test-bucket/models/model.tar.gz'
            
            response = deployment.create_sagemaker_model(
                model_name=model_name,
                model_data_url=model_data,
                execution_role='arn:aws:iam::123456789:role/SageMakerRole',
                container_image='sklearn-image:latest'
            )
            
            assert 'ModelArn' in response
            mock_sm.create_model.assert_called_once()
    
    def test_endpoint_configuration_generation(self):
        """Test endpoint configuration creation"""
        deployment = SageMakerDeployment()
        config = AWSConfig()
        
        endpoint_config = deployment.generate_endpoint_config(
            model_name='test-model',
            instance_type=config.ENDPOINT_INSTANCE_SMALL,
            instance_count=1
        )
        
        assert 'ProductionVariants' in endpoint_config
        assert endpoint_config['ProductionVariants'][0]['InstanceType'] == config.ENDPOINT_INSTANCE_SMALL
        assert endpoint_config['ProductionVariants'][0]['InitialInstanceCount'] == 1
    
    def test_blue_green_deployment_config(self):
        """Test blue/green deployment configuration"""
        deployment = SageMakerDeployment()
        
        config = deployment.create_blue_green_config(
            blue_model='model-v1',
            green_model='model-v2',
            traffic_split={'blue': 90, 'green': 10}
        )
        
        assert len(config['ProductionVariants']) == 2
        assert config['ProductionVariants'][0]['InitialVariantWeight'] == 90
        assert config['ProductionVariants'][1]['InitialVariantWeight'] == 10
    
    @pytest.mark.slow
    def test_endpoint_deployment_workflow(self, mock_sagemaker_client):
        """Test complete endpoint deployment workflow"""
        with patch('boto3.client', return_value=mock_sagemaker_client):
            deployment = SageMakerDeployment()
            
            # Mock successful deployment
            mock_sagemaker_client.describe_endpoint.return_value = {
                'EndpointStatus': 'InService',
                'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789:endpoint/test'
            }
            
            status = deployment.wait_for_endpoint('test-endpoint', timeout=10)
            
            assert status == 'InService'


@pytest.mark.integration
class TestModelMonitoring:
    """Test model monitoring setup"""
    
    @mock_s3
    def test_baseline_dataset_creation(self, sample_fraud_data):
        """Test creating baseline dataset for monitoring"""
        from src.ml_toolkit.sagemaker_utils import ModelMonitor
        
        # Create S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-monitoring-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        monitor = ModelMonitor(bucket_name=bucket_name)
        
        # Create baseline
        baseline_uri = monitor.create_baseline(
            data=sample_fraud_data,
            prefix='baselines/fraud-detection'
        )
        
        assert baseline_uri.startswith(f's3://{bucket_name}/baselines')
        
        # Verify file uploaded
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='baselines')
        assert 'Contents' in objects
    
    def test_drift_detection_calculation(self, sample_fraud_data):
        """Test drift detection logic"""
        from src.ml_toolkit.sagemaker_utils import ModelMonitor
        from scipy.stats import chi2_contingency
        
        monitor = ModelMonitor()
        
        # Create baseline distribution
        baseline = sample_fraud_data['merchant_category'].value_counts(normalize=True)
        
        # Create drifted distribution (more fraud patterns)
        drifted_data = sample_fraud_data.copy()
        drifted_data = drifted_data[drifted_data['is_fraud'] == 1].sample(frac=0.5, replace=True)
        drifted = drifted_data['merchant_category'].value_counts(normalize=True)
        
        # Align indices
        baseline = baseline.reindex(drifted.index, fill_value=0)
        
        # Calculate drift
        chi2, p_value = monitor.calculate_distribution_drift(baseline, drifted)
        
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # With significant distribution change, should detect drift
        if p_value < 0.05:
            print(f"✓ Drift detected (p-value: {p_value:.4f})")
    
    def test_monitoring_schedule_creation(self):
        """Test monitoring schedule configuration"""
        from src.ml_toolkit.sagemaker_utils import ModelMonitor
        
        monitor = ModelMonitor()
        config = AWSConfig()
        
        schedule_config = monitor.create_monitoring_schedule(
            endpoint_name='test-endpoint',
            schedule_expression=config.MONITORING_SCHEDULE_CRON,
            baseline_uri='s3://bucket/baseline.csv'
        )
        
        assert 'ScheduleExpression' in schedule_config
        assert schedule_config['ScheduleExpression'] == config.MONITORING_SCHEDULE_CRON


@pytest.mark.integration
class TestFeatureStore:
    """Test Feature Store operations"""
    
    def test_feature_group_creation_config(self):
        """Test feature group configuration"""
        from src.ml_toolkit.sagemaker_utils import FeatureStoreManager
        
        manager = FeatureStoreManager()
        
        feature_definitions = [
            {'FeatureName': 'customer_id', 'FeatureType': 'String'},
            {'FeatureName': 'tenure', 'FeatureType': 'Integral'},
            {'FeatureName': 'monthly_charges', 'FeatureType': 'Fractional'},
        ]
        
        config = manager.create_feature_group_config(
            group_name='customer-features',
            record_identifier='customer_id',
            event_time_feature='event_time',
            feature_definitions=feature_definitions
        )
        
        assert config['FeatureGroupName'] == 'customer-features'
        assert config['RecordIdentifierFeatureName'] == 'customer_id'
        assert len(config['FeatureDefinitions']) == 3
    
    def test_feature_ingestion_format(self, sample_churn_data):
        """Test feature ingestion data formatting"""
        from src.ml_toolkit.sagemaker_utils import FeatureStoreManager
        import time
        
        manager = FeatureStoreManager()
        
        # Add required columns
        sample_churn_data['customer_id'] = [f'CUST_{i:05d}' for i in range(len(sample_churn_data))]
        sample_churn_data['event_time'] = time.time()
        
        records = manager.format_records_for_ingestion(sample_churn_data)
        
        assert len(records) == len(sample_churn_data)
        assert all('customer_id' in record for record in records)
        assert all('event_time' in record for record in records)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """Test complete ML pipeline from training to deployment"""
    
    def test_fraud_detection_pipeline(self, sample_fraud_data, temp_model_dir):
        """Test fraud detection end-to-end pipeline"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from src.ml_toolkit.preprocessing import DataPreprocessor
        from src.ml_toolkit.evaluation import ModelEvaluator
        
        # Prepare data
        preprocessor = DataPreprocessor()
        X = sample_fraud_data.drop('is_fraud', axis=1)
        y = sample_fraud_data['is_fraud']
        
        # Encode categorical
        X_encoded = preprocessor.encode_categorical(X, ['merchant_category'])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Train
        model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Evaluate
        evaluator = ModelEvaluator()
        y_pred = model.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred, average='binary')
        
        # Assert reasonable performance
        assert metrics['accuracy'] > 0.5  # Better than random
        assert metrics['f1'] > 0.4
        
        # Save model
        import joblib
        model_path = temp_model_dir / "fraud_model.pkl"
        joblib.dump(model, model_path)
        
        assert model_path.exists()
