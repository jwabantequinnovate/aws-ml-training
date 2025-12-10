"""
SageMaker Debugger Integration

Monitor training jobs in real-time with automatic rules and custom hooks.
"""

from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, CollectionConfig
from sagemaker.estimator import Estimator
import sagemaker


def create_debugger_config():
    """
    Configure SageMaker Debugger with common rules
    
    Debugger automatically detects issues during training:
    - Vanishing/exploding gradients
    - Overfitting
    - Poor weight initialization
    - Loss not decreasing
    """
    
    # Save tensors every 100 steps
    debugger_hook_config = DebuggerHookConfig(
        s3_output_path=f"s3://{sagemaker.Session().default_bucket()}/debugger",
        collection_configs=[
            CollectionConfig(
                name="losses",
                parameters={"save_interval": "100"}
            ),
            CollectionConfig(
                name="weights",
                parameters={"save_interval": "500"}
            ),
            CollectionConfig(
                name="gradients",
                parameters={"save_interval": "500"}
            ),
        ]
    )
    
    # Built-in rules for common issues
    rules = [
        # Detect if loss is not decreasing
        Rule.sagemaker(
            rule_configs.loss_not_decreasing(),
            rule_parameters={
                "tensor_regex": ".*loss.*",
                "patience": "10",
                "min_delta": "0.01"
            }
        ),
        
        # Detect overfitting
        Rule.sagemaker(
            rule_configs.overfit(),
            rule_parameters={
                "patience": "5",
                "ratio_threshold": "0.1"
            }
        ),
        
        # Detect vanishing gradients
        Rule.sagemaker(
            rule_configs.vanishing_gradient(),
            rule_parameters={
                "threshold": "0.0000001"
            }
        ),
        
        # Detect exploding tensors
        Rule.sagemaker(
            rule_configs.exploding_tensor(),
            rule_parameters={
                "tensor_regex": ".*gradient.*",
                "only_nan": "False"
            }
        ),
        
        # Check weight initialization
        Rule.sagemaker(
            rule_configs.poor_weight_initialization(),
            rule_parameters={
                "activation_inputs_regex": ".*relu_input|.*tanh_input",
                "threshold": "0.2"
            }
        ),
    ]
    
    return debugger_hook_config, rules


def create_estimator_with_debugger(
    role,
    instance_type='ml.m5.xlarge',
    image_uri=None,
    hyperparameters=None
):
    """
    Create estimator with Debugger enabled
    
    Example usage:
        estimator = create_estimator_with_debugger(
            role=sagemaker.get_execution_role(),
            hyperparameters={'epochs': 10, 'learning_rate': 0.01}
        )
        estimator.fit({'training': 's3://bucket/data'})
    """
    
    debugger_hook_config, rules = create_debugger_config()
    
    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        
        # Enable Debugger
        debugger_hook_config=debugger_hook_config,
        rules=rules,
        
        # Profiler for system metrics (CPU, GPU, memory)
        profiler_config=None,  # Disabled to save costs, enable in production
    )
    
    return estimator


def check_training_issues(training_job_name):
    """
    Check if Debugger detected any issues during training
    
    Returns:
        dict: Summary of detected issues
    """
    import boto3
    
    sagemaker_client = boto3.client('sagemaker')
    
    # Get training job details
    response = sagemaker_client.describe_training_job(
        TrainingJobName=training_job_name
    )
    
    # Check debugger rule evaluations
    rule_statuses = response.get('DebugRuleEvaluationStatuses', [])
    
    issues = {
        'has_issues': False,
        'rules_triggered': [],
        'all_rules': []
    }
    
    for rule in rule_statuses:
        rule_name = rule['RuleConfigurationName']
        rule_status = rule['RuleEvaluationStatus']
        
        issues['all_rules'].append({
            'name': rule_name,
            'status': rule_status
        })
        
        if rule_status == 'IssuesFound':
            issues['has_issues'] = True
            issues['rules_triggered'].append(rule_name)
    
    return issues


# Example: Custom Debugger Hook
def create_custom_hook():
    """
    Custom hook for specific monitoring needs
    
    Use this when built-in rules don't cover your use case.
    """
    
    custom_hook_config = DebuggerHookConfig(
        s3_output_path=f"s3://{sagemaker.Session().default_bucket()}/custom-debugger",
        collection_configs=[
            # Monitor specific layers
            CollectionConfig(
                name="custom_collection",
                parameters={
                    "include_regex": ".*dense.*|.*conv.*",  # Only dense and conv layers
                    "save_interval": "100",
                    "save_steps": "0,100,200,300,400,500"  # Specific steps
                }
            )
        ]
    )
    
    return custom_hook_config


# Example: Real-time monitoring script
def monitor_training_realtime(training_job_name, check_interval=60):
    """
    Monitor training job in real-time
    
    Args:
        training_job_name: Name of the training job
        check_interval: How often to check (seconds)
    """
    import boto3
    import time
    
    sagemaker_client = boto3.client('sagemaker')
    
    print(f"🔍 Monitoring training job: {training_job_name}")
    print(f"Checking every {check_interval} seconds...\n")
    
    while True:
        response = sagemaker_client.describe_training_job(
            TrainingJobName=training_job_name
        )
        
        status = response['TrainingJobStatus']
        print(f"Status: {status}")
        
        if status in ['Completed', 'Failed', 'Stopped']:
            break
        
        # Check debugger rules
        rule_statuses = response.get('DebugRuleEvaluationStatuses', [])
        for rule in rule_statuses:
            rule_name = rule['RuleConfigurationName']
            rule_status = rule['RuleEvaluationStatus']
            
            if rule_status == 'IssuesFound':
                print(f"⚠️  {rule_name}: Issues detected!")
            elif rule_status == 'InProgress':
                print(f"🔄 {rule_name}: Monitoring...")
        
        time.sleep(check_interval)
    
    print("\n✅ Training completed!")
    
    # Final issues check
    issues = check_training_issues(training_job_name)
    if issues['has_issues']:
        print("\n⚠️  Issues found during training:")
        for rule in issues['rules_triggered']:
            print(f"   - {rule}")
    else:
        print("✅ No issues detected")
