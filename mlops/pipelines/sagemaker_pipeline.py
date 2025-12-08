"""
SageMaker Pipeline for MLOps

This module demonstrates how to create an end-to-end ML pipeline using
SageMaker Pipelines for automated training, evaluation, and deployment.
"""

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep


class MLOpsPipeline:
    """
    Create and manage SageMaker Pipelines for MLOps
    """
    
    def __init__(self, pipeline_name: str, role: str = None):
        """
        Initialize MLOps Pipeline
        
        Args:
            pipeline_name: Name for the pipeline
            role: IAM role for SageMaker
        """
        self.pipeline_name = pipeline_name
        self.session = sagemaker.Session()
        self.region = boto3.Session().region_name
        self.bucket = self.session.default_bucket()
        
        if role is None:
            self.role = sagemaker.get_execution_role()
        else:
            self.role = role
    
    def create_parameters(self) -> dict:
        """
        Create pipeline parameters
        
        Returns:
            Dictionary of parameters
        """
        parameters = {
            'input_data': ParameterString(
                name='InputData',
                default_value=f's3://{self.bucket}/input-data'
            ),
            'instance_type': ParameterString(
                name='InstanceType',
                default_value='ml.m5.xlarge'
            ),
            'instance_count': ParameterInteger(
                name='InstanceCount',
                default_value=1
            ),
            'model_approval_threshold': ParameterFloat(
                name='ModelApprovalThreshold',
                default_value=0.85
            )
        }
        return parameters
    
    def create_preprocessing_step(
        self,
        parameters: dict,
        processing_script: str
    ) -> ProcessingStep:
        """
        Create data preprocessing step
        
        Args:
            parameters: Pipeline parameters
            processing_script: Path to preprocessing script
            
        Returns:
            ProcessingStep
        """
        sklearn_processor = SKLearnProcessor(
            framework_version='1.2-1',
            role=self.role,
            instance_type=parameters['instance_type'],
            instance_count=1,
            base_job_name='preprocessing'
        )
        
        step_process = ProcessingStep(
            name='PreprocessData',
            processor=sklearn_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=parameters['input_data'],
                    destination='/opt/ml/processing/input'
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name='train',
                    source='/opt/ml/processing/train',
                    destination=f's3://{self.bucket}/processed/train'
                ),
                sagemaker.processing.ProcessingOutput(
                    output_name='validation',
                    source='/opt/ml/processing/validation',
                    destination=f's3://{self.bucket}/processed/validation'
                ),
                sagemaker.processing.ProcessingOutput(
                    output_name='test',
                    source='/opt/ml/processing/test',
                    destination=f's3://{self.bucket}/processed/test'
                )
            ],
            code=processing_script
        )
        
        return step_process
    
    def create_training_step(
        self,
        parameters: dict,
        step_process: ProcessingStep,
        training_script: str
    ) -> TrainingStep:
        """
        Create model training step
        
        Args:
            parameters: Pipeline parameters
            step_process: Preprocessing step
            training_script: Path to training script
            
        Returns:
            TrainingStep
        """
        estimator = Estimator(
            image_uri=sagemaker.image_uris.retrieve(
                framework='sklearn',
                region=self.region,
                version='1.2-1'
            ),
            role=self.role,
            instance_count=parameters['instance_count'],
            instance_type=parameters['instance_type'],
            output_path=f's3://{self.bucket}/models',
            base_job_name='training'
        )
        
        estimator.set_hyperparameters({
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'num_round': 100
        })
        
        step_train = TrainingStep(
            name='TrainModel',
            estimator=estimator,
            inputs={
                'train': TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                    content_type='text/csv'
                ),
                'validation': TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                    content_type='text/csv'
                )
            }
        )
        
        return step_train
    
    def create_evaluation_step(
        self,
        step_train: TrainingStep,
        step_process: ProcessingStep,
        evaluation_script: str
    ) -> ProcessingStep:
        """
        Create model evaluation step
        
        Args:
            step_train: Training step
            step_process: Preprocessing step
            evaluation_script: Path to evaluation script
            
        Returns:
            ProcessingStep for evaluation
        """
        sklearn_processor = SKLearnProcessor(
            framework_version='1.2-1',
            role=self.role,
            instance_type='ml.m5.xlarge',
            instance_count=1,
            base_job_name='evaluation'
        )
        
        evaluation_report = PropertyFile(
            name='EvaluationReport',
            output_name='evaluation',
            path='evaluation.json'
        )
        
        step_eval = ProcessingStep(
            name='EvaluateModel',
            processor=sklearn_processor,
            inputs=[
                sagemaker.processing.ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination='/opt/ml/processing/model'
                ),
                sagemaker.processing.ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                    destination='/opt/ml/processing/test'
                )
            ],
            outputs=[
                sagemaker.processing.ProcessingOutput(
                    output_name='evaluation',
                    source='/opt/ml/processing/evaluation',
                    destination=f's3://{self.bucket}/evaluation'
                )
            ],
            code=evaluation_script,
            property_files=[evaluation_report]
        )
        
        return step_eval, evaluation_report
    
    def create_conditional_step(
        self,
        parameters: dict,
        step_eval: ProcessingStep,
        evaluation_report: PropertyFile,
        step_train: TrainingStep
    ) -> ConditionStep:
        """
        Create conditional step for model approval
        
        Args:
            parameters: Pipeline parameters
            step_eval: Evaluation step
            evaluation_report: Evaluation property file
            step_train: Training step
            
        Returns:
            ConditionStep
        """
        # Create model
        model = Model(
            image_uri=sagemaker.image_uris.retrieve(
                framework='sklearn',
                region=self.region,
                version='1.2-1'
            ),
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=self.session,
            role=self.role
        )
        
        step_create_model = CreateModelStep(
            name='CreateModel',
            model=model,
            inputs=sagemaker.inputs.CreateModelInput(
                instance_type='ml.m5.xlarge'
            )
        )
        
        # Condition: accuracy >= threshold
        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_eval.name,
                property_file=evaluation_report,
                json_path='metrics.accuracy.value'
            ),
            right=parameters['model_approval_threshold']
        )
        
        step_cond = ConditionStep(
            name='CheckAccuracy',
            conditions=[cond_gte],
            if_steps=[step_create_model],
            else_steps=[]
        )
        
        return step_cond
    
    def create_pipeline(
        self,
        processing_script: str,
        training_script: str,
        evaluation_script: str
    ) -> Pipeline:
        """
        Create complete MLOps pipeline
        
        Args:
            processing_script: Path to preprocessing script
            training_script: Path to training script
            evaluation_script: Path to evaluation script
            
        Returns:
            Pipeline object
        """
        # Create parameters
        parameters = self.create_parameters()
        
        # Create steps
        step_process = self.create_preprocessing_step(parameters, processing_script)
        step_train = self.create_training_step(parameters, step_process, training_script)
        step_eval, evaluation_report = self.create_evaluation_step(
            step_train, step_process, evaluation_script
        )
        step_cond = self.create_conditional_step(
            parameters, step_eval, evaluation_report, step_train
        )
        
        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                parameters['input_data'],
                parameters['instance_type'],
                parameters['instance_count'],
                parameters['model_approval_threshold']
            ],
            steps=[step_process, step_train, step_eval, step_cond]
        )
        
        return pipeline
    
    def upsert_pipeline(
        self,
        processing_script: str,
        training_script: str,
        evaluation_script: str
    ) -> dict:
        """
        Create or update pipeline
        
        Args:
            processing_script: Path to preprocessing script
            training_script: Path to training script
            evaluation_script: Path to evaluation script
            
        Returns:
            Pipeline ARN
        """
        pipeline = self.create_pipeline(
            processing_script, training_script, evaluation_script
        )
        
        response = pipeline.upsert(role_arn=self.role)
        print(f"✓ Pipeline upserted: {self.pipeline_name}")
        print(f"  ARN: {response['PipelineArn']}")
        
        return response
    
    def execute_pipeline(
        self,
        parameters: dict = None
    ) -> str:
        """
        Execute the pipeline
        
        Args:
            parameters: Pipeline execution parameters
            
        Returns:
            Execution ARN
        """
        pipeline = Pipeline(name=self.pipeline_name)
        
        execution = pipeline.start(parameters=parameters)
        print(f"✓ Pipeline execution started")
        print(f"  Execution ARN: {execution.arn}")
        
        return execution.arn
    
    def describe_execution(
        self,
        execution_arn: str
    ) -> dict:
        """
        Get pipeline execution details
        
        Args:
            execution_arn: Execution ARN
            
        Returns:
            Execution details
        """
        client = boto3.client('sagemaker')
        response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )
        return response


def example_usage():
    """
    Example usage of MLOpsPipeline
    """
    # Create pipeline
    pipeline = MLOpsPipeline(
        pipeline_name='fraud-detection-pipeline'
    )
    
    # Create/update pipeline
    response = pipeline.upsert_pipeline(
        processing_script='scripts/preprocessing.py',
        training_script='scripts/train.py',
        evaluation_script='scripts/evaluate.py'
    )
    
    # Execute pipeline
    execution_arn = pipeline.execute_pipeline(
        parameters={
            'InputData': 's3://bucket/raw-data',
            'InstanceType': 'ml.m5.2xlarge',
            'ModelApprovalThreshold': 0.90
        }
    )
    
    print(f"\n✓ Pipeline workflow complete!")
    print(f"Monitor execution at:")
    print(f"https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/pipelines")


if __name__ == '__main__':
    example_usage()
