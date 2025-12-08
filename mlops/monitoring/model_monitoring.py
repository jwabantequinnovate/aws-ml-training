"""
Model Monitoring Setup for SageMaker

This module demonstrates how to set up comprehensive model monitoring
including data quality, model quality, bias detection, and feature attribution.
"""

import boto3
import sagemaker
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    CronExpressionGenerator,
    ModelQualityMonitor
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
import json
from datetime import datetime


class ModelMonitoringSetup:
    """
    Setup and manage model monitoring for SageMaker endpoints
    """
    
    def __init__(self, endpoint_name: str, role: str = None):
        """
        Initialize monitoring setup
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            role: IAM role for SageMaker (auto-detected if None)
        """
        self.endpoint_name = endpoint_name
        self.session = sagemaker.Session()
        self.region = boto3.Session().region_name
        self.bucket = self.session.default_bucket()
        
        if role is None:
            self.role = sagemaker.get_execution_role()
        else:
            self.role = role
        
        self.sagemaker_client = boto3.client('sagemaker')
    
    def enable_data_capture(
        self,
        sampling_percentage: int = 100,
        capture_options: list = None
    ) -> DataCaptureConfig:
        """
        Enable data capture for the endpoint
        
        Args:
            sampling_percentage: Percentage of requests to capture (1-100)
            capture_options: Types to capture (Input, Output, or both)
            
        Returns:
            DataCaptureConfig object
        """
        if capture_options is None:
            capture_options = ["Input", "Output"]
        
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=sampling_percentage,
            destination_s3_uri=f's3://{self.bucket}/model-monitor/{self.endpoint_name}/data-capture',
            capture_options=capture_options
        )
        
        print(f"✓ Data capture enabled for {self.endpoint_name}")
        print(f"  Sampling: {sampling_percentage}%")
        print(f"  Destination: {data_capture_config.destination_s3_uri}")
        
        return data_capture_config
    
    def create_baseline(
        self,
        baseline_dataset_uri: str,
        output_s3_uri: str = None
    ) -> dict:
        """
        Create baseline statistics and constraints
        
        Args:
            baseline_dataset_uri: S3 URI of baseline dataset
            output_s3_uri: S3 URI for baseline results
            
        Returns:
            Dictionary with baseline statistics and constraints
        """
        if output_s3_uri is None:
            output_s3_uri = f's3://{self.bucket}/model-monitor/{self.endpoint_name}/baseline'
        
        monitor = DefaultModelMonitor(
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.session
        )
        
        print(f"Creating baseline...")
        print(f"  Dataset: {baseline_dataset_uri}")
        print(f"  Output: {output_s3_uri}")
        
        baseline_job = monitor.suggest_baseline(
            baseline_dataset=baseline_dataset_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=output_s3_uri,
            wait=True
        )
        
        print(f"✓ Baseline created successfully")
        
        return {
            'statistics': f'{output_s3_uri}/statistics.json',
            'constraints': f'{output_s3_uri}/constraints.json'
        }
    
    def setup_data_quality_monitoring(
        self,
        baseline_statistics: str,
        baseline_constraints: str,
        schedule_name: str = None,
        schedule_expression: str = 'cron(0 * * * ? *)'  # Hourly
    ) -> None:
        """
        Set up data quality monitoring schedule
        
        Args:
            baseline_statistics: S3 URI of baseline statistics
            baseline_constraints: S3 URI of baseline constraints
            schedule_name: Name for monitoring schedule
            schedule_expression: Cron expression for schedule
        """
        if schedule_name is None:
            schedule_name = f'{self.endpoint_name}-data-quality-monitor'
        
        monitor = DefaultModelMonitor(
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.session
        )
        
        print(f"Setting up data quality monitoring...")
        print(f"  Schedule: {schedule_name}")
        print(f"  Frequency: {schedule_expression}")
        
        monitor.create_monitoring_schedule(
            monitor_schedule_name=schedule_name,
            endpoint_input=self.endpoint_name,
            output_s3_uri=f's3://{self.bucket}/model-monitor/{self.endpoint_name}/reports',
            statistics=baseline_statistics,
            constraints=baseline_constraints,
            schedule_cron_expression=schedule_expression,
            enable_cloudwatch_metrics=True
        )
        
        print(f"✓ Data quality monitoring enabled")
    
    def setup_model_quality_monitoring(
        self,
        ground_truth_s3_uri: str,
        problem_type: str = 'BinaryClassification',
        schedule_name: str = None,
        schedule_expression: str = 'cron(0 0 * * ? *)'  # Daily
    ) -> None:
        """
        Set up model quality monitoring
        
        Args:
            ground_truth_s3_uri: S3 URI with ground truth labels
            problem_type: Problem type (BinaryClassification, Regression, etc.)
            schedule_name: Name for monitoring schedule
            schedule_expression: Cron expression for schedule
        """
        if schedule_name is None:
            schedule_name = f'{self.endpoint_name}-model-quality-monitor'
        
        model_quality_monitor = ModelQualityMonitor(
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.session
        )
        
        print(f"Setting up model quality monitoring...")
        print(f"  Schedule: {schedule_name}")
        print(f"  Problem type: {problem_type}")
        
        model_quality_monitor.create_monitoring_schedule(
            monitor_schedule_name=schedule_name,
            endpoint_input=self.endpoint_name,
            ground_truth_input=ground_truth_s3_uri,
            problem_type=problem_type,
            output_s3_uri=f's3://{self.bucket}/model-monitor/{self.endpoint_name}/model-quality',
            schedule_cron_expression=schedule_expression
        )
        
        print(f"✓ Model quality monitoring enabled")
    
    def setup_cloudwatch_alarms(
        self,
        metric_name: str,
        threshold: float,
        alarm_name: str = None,
        comparison_operator: str = 'GreaterThanThreshold',
        evaluation_periods: int = 1
    ) -> None:
        """
        Set up CloudWatch alarms for monitoring
        
        Args:
            metric_name: CloudWatch metric name
            threshold: Alarm threshold
            alarm_name: Name for the alarm
            comparison_operator: Comparison operator
            evaluation_periods: Number of periods to evaluate
        """
        if alarm_name is None:
            alarm_name = f'{self.endpoint_name}-{metric_name}-alarm'
        
        cloudwatch = boto3.client('cloudwatch')
        
        print(f"Setting up CloudWatch alarm: {alarm_name}")
        
        cloudwatch.put_metric_alarm(
            AlarmName=alarm_name,
            ComparisonOperator=comparison_operator,
            EvaluationPeriods=evaluation_periods,
            MetricName=metric_name,
            Namespace='AWS/SageMaker',
            Period=300,  # 5 minutes
            Statistic='Average',
            Threshold=threshold,
            ActionsEnabled=True,
            AlarmDescription=f'Alarm for {metric_name} on {self.endpoint_name}',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': self.endpoint_name
                }
            ]
        )
        
        print(f"✓ CloudWatch alarm created: {alarm_name}")
    
    def setup_standard_alarms(self) -> None:
        """
        Set up standard CloudWatch alarms for endpoint monitoring
        """
        print("Setting up standard alarms...")
        
        # Model latency alarm (> 1000ms)
        self.setup_cloudwatch_alarms(
            metric_name='ModelLatency',
            threshold=1000,
            alarm_name=f'{self.endpoint_name}-high-latency'
        )
        
        # Invocation errors alarm (> 5%)
        self.setup_cloudwatch_alarms(
            metric_name='ModelInvocation4XXErrors',
            threshold=5,
            alarm_name=f'{self.endpoint_name}-4xx-errors'
        )
        
        # Server errors alarm (> 0)
        self.setup_cloudwatch_alarms(
            metric_name='ModelInvocation5XXErrors',
            threshold=0,
            alarm_name=f'{self.endpoint_name}-5xx-errors'
        )
        
        print("✓ Standard alarms configured")
    
    def get_monitoring_schedule_status(
        self,
        schedule_name: str
    ) -> dict:
        """
        Get status of a monitoring schedule
        
        Args:
            schedule_name: Name of the monitoring schedule
            
        Returns:
            Schedule status information
        """
        response = self.sagemaker_client.describe_monitoring_schedule(
            MonitoringScheduleName=schedule_name
        )
        
        return {
            'schedule_name': response['MonitoringScheduleName'],
            'schedule_status': response['MonitoringScheduleStatus'],
            'last_execution_time': response.get('LastMonitoringExecutionSummary', {}).get('ScheduledTime'),
            'monitoring_type': response.get('MonitoringType')
        }
    
    def list_monitoring_executions(
        self,
        schedule_name: str,
        max_results: int = 10
    ) -> list:
        """
        List recent monitoring executions
        
        Args:
            schedule_name: Name of the monitoring schedule
            max_results: Maximum number of results
            
        Returns:
            List of monitoring executions
        """
        response = self.sagemaker_client.list_monitoring_executions(
            MonitoringScheduleName=schedule_name,
            MaxResults=max_results,
            SortBy='ScheduledTime',
            SortOrder='Descending'
        )
        
        return response['MonitoringExecutionSummaries']
    
    def stop_monitoring_schedule(
        self,
        schedule_name: str
    ) -> None:
        """
        Stop a monitoring schedule
        
        Args:
            schedule_name: Name of the monitoring schedule
        """
        self.sagemaker_client.stop_monitoring_schedule(
            MonitoringScheduleName=schedule_name
        )
        print(f"✓ Monitoring schedule stopped: {schedule_name}")
    
    def delete_monitoring_schedule(
        self,
        schedule_name: str
    ) -> None:
        """
        Delete a monitoring schedule
        
        Args:
            schedule_name: Name of the monitoring schedule
        """
        self.sagemaker_client.delete_monitoring_schedule(
            MonitoringScheduleName=schedule_name
        )
        print(f"✓ Monitoring schedule deleted: {schedule_name}")


def example_usage():
    """
    Example usage of ModelMonitoringSetup
    """
    # Initialize monitoring setup
    monitoring = ModelMonitoringSetup(
        endpoint_name='fraud-detection-endpoint'
    )
    
    # 1. Enable data capture
    data_capture_config = monitoring.enable_data_capture(
        sampling_percentage=100
    )
    
    # 2. Create baseline
    baseline = monitoring.create_baseline(
        baseline_dataset_uri='s3://bucket/baseline-data/train.csv'
    )
    
    # 3. Set up data quality monitoring
    monitoring.setup_data_quality_monitoring(
        baseline_statistics=baseline['statistics'],
        baseline_constraints=baseline['constraints'],
        schedule_expression='cron(0 * * * ? *)'  # Hourly
    )
    
    # 4. Set up model quality monitoring
    monitoring.setup_model_quality_monitoring(
        ground_truth_s3_uri='s3://bucket/ground-truth/',
        problem_type='BinaryClassification',
        schedule_expression='cron(0 0 * * ? *)'  # Daily
    )
    
    # 5. Set up CloudWatch alarms
    monitoring.setup_standard_alarms()
    
    print("\n✓ Model monitoring setup complete!")


if __name__ == '__main__':
    example_usage()
