# Lab 9: Amazon Bedrock - GenAI Model Comparison & Security

## Overview

This lab introduces Amazon Bedrock for generative AI workloads, covering model selection, security best practices, and integration with SageMaker MLOps workflows.

## Learning Objectives

By the end of this lab, you will be able to:
- Compare different foundation models available in Bedrock
- Implement secure API access with IAM and guardrails
- Evaluate model performance for specific use cases
- Integrate Bedrock with existing ML pipelines
- Apply content filtering and safety controls

## Prerequisites

- AWS account with Bedrock access enabled
- Completed Labs 5-6 (Model Registry, Endpoints)
- Understanding of LLM concepts
- IAM permissions for Bedrock service

## Estimated Duration

60 minutes

## Topics Covered

1. **Foundation Model Selection** (15 min)
   - Claude, Titan, Llama model families
   - Model capabilities comparison
   - Cost and performance trade-offs

2. **Security & Guardrails** (20 min)
   - IAM policies for Bedrock access
   - Content filtering and moderation
   - PII detection and redaction
   - Prompt injection protection

3. **Model Evaluation** (15 min)
   - Performance benchmarking
   - Quality assessment metrics
   - Cost analysis per use case

4. **MLOps Integration** (10 min)
   - Bedrock + SageMaker workflows
   - Model versioning and tracking
   - Monitoring and observability

## AWS Services Used

- Amazon Bedrock
- AWS IAM
- Amazon CloudWatch
- AWS Secrets Manager (optional)

## Key Concepts

- **Foundation Models**: Pre-trained large models (Claude, Titan, etc.)
- **Guardrails**: Security controls for content filtering
- **Model Invocation**: API-based inference without deployment
- **Responsible AI**: Safety, bias detection, content moderation
