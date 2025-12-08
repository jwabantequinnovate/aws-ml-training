# Instructor Guide - AWS ML Training

## Overview

This guide provides instructors with comprehensive information for delivering the AWS Machine Learning training program to senior engineers.

## Program Structure

### Total Duration: 4-5 Days (28-42 hours)

**Day 1-2: Foundation & Use Cases**
- Module 1: Fraud Detection (4-6 hours)
- Module 2: Customer Churn (4-6 hours)

**Day 3-4: Advanced Topics**
- Module 3: Text Classification (5-7 hours)
- Module 4: Sentiment Analysis (4-6 hours)

**Day 5: MLOps & Architecture**
- MLOps Components (4-6 hours)
- Architecture Review (2-4 hours)
- Q&A and Wrap-up

## Teaching Methodology

### Hands-on Learning (70%)
- Students work through exercises independently
- Instructor available for guidance
- Pair programming encouraged

### Lectures & Discussions (20%)
- Concept introduction (15 min)
- Solution review (30 min)
- Architecture discussions (45 min)

### Q&A and Troubleshooting (10%)
- Individual support
- Group discussions
- Real-world scenarios

## Module-by-Module Guide

### Module 1: Fraud Detection

**Learning Objectives:**
- Handle imbalanced datasets
- Build fraud detection models
- Deploy real-time inference endpoints

**Pre-session Setup:**
- Verify all students have AWS access
- Ensure SageMaker permissions are configured
- Test notebook environment

**Lecture Topics (45 min):**
1. Introduction to fraud detection (10 min)
2. Handling imbalanced data (15 min)
3. Model evaluation for fraud (10 min)
4. Real-time deployment (10 min)

**Exercise Time: 2-3 hours**
- Monitor student progress
- Common issues to watch for:
  - SMOTE implementation
  - Feature scaling
  - Threshold selection

**Solution Review (30 min):**
- Walk through complete solution
- Discuss alternative approaches
- Emphasize best practices

**Discussion Questions:**
1. How to handle concept drift?
2. Business cost of false positives vs false negatives
3. Production monitoring strategies

**Key Takeaways:**
- Class imbalance requires special handling
- Feature engineering is critical
- Explainability matters for compliance

### Module 2: Customer Churn

**Learning Objectives:**
- Feature engineering for customer data
- Model comparison frameworks
- Batch inference patterns

**Lecture Topics (45 min):**
1. Customer churn analytics (10 min)
2. Ensemble methods (15 min)
3. Batch transform (15 min)
4. Model registry (5 min)

**Exercise Time: 2-3 hours**
- Focus on feature engineering quality
- Ensure proper model comparison
- Guide batch transform setup

**Common Challenges:**
- Temporal feature creation
- Handling categorical variables
- Batch job configuration

**Discussion Questions:**
1. When to retrain models?
2. How to define churn?
3. Cost optimization for batch inference

**Key Takeaways:**
- Systematic model comparison is essential
- Batch inference for periodic predictions
- Business metrics drive decisions

### Module 3: Text Classification

**Learning Objectives:**
- NLP preprocessing
- Fine-tune transformer models
- Async inference deployment

**Lecture Topics (60 min):**
1. Text preprocessing (15 min)
2. Transfer learning & BERT (20 min)
3. Multi-class classification (15 min)
4. Async inference (10 min)

**Exercise Time: 3-4 hours**
- BERT fine-tuning can be time-consuming
- Monitor GPU usage and costs
- Ensure proper tokenization

**Advanced Topics:**
- Model compression
- Inference optimization
- Multi-lingual support

**Discussion Questions:**
1. When to use pre-trained models?
2. How to handle domain-specific text?
3. Balancing accuracy vs latency

**Key Takeaways:**
- Transfer learning accelerates development
- Async inference for large payloads
- Model size vs performance trade-offs

### Module 4: Sentiment Analysis

**Learning Objectives:**
- Sentiment analysis pipelines
- Multi-lingual models
- A/B testing deployments

**Lecture Topics (45 min):**
1. Sentiment analysis approaches (15 min)
2. Hugging Face ecosystem (15 min)
3. A/B testing & canary deployment (15 min)

**Exercise Time: 2-3 hours**
- Guide model selection from HuggingFace
- Implement A/B testing carefully
- Monitor variant performance

**Demonstration:**
- Live A/B testing setup
- Traffic splitting
- Monitoring dashboards

**Discussion Questions:**
1. How to measure A/B test success?
2. When to promote new model?
3. Handling multi-lingual content

**Key Takeaways:**
- Pre-trained models save development time
- A/B testing reduces deployment risk
- Monitor business metrics, not just accuracy

## MLOps Deep Dive

**Duration: 4-6 hours**

**Topics to Cover:**

### 1. Deployment Strategies (90 min)
- Real-time inference
- Batch transform
- Async inference
- When to use each

**Interactive Exercise:**
- Students choose deployment for a scenario
- Group discussion on trade-offs

### 2. Model Registry & Versioning (60 min)
- SageMaker Model Registry
- Model lineage tracking
- Version management

**Demo:**
- Register models
- Promote to production
- Rollback scenario

### 3. Monitoring & Observability (90 min)
- CloudWatch metrics
- Model Monitor for data drift
- Performance tracking
- Alerting setup

**Hands-on:**
- Set up monitoring dashboard
- Configure alerts
- Simulate drift detection

### 4. CI/CD Pipelines (60 min)
- CodeBuild setup
- Automated testing
- Deployment automation
- Jenkins integration

**Activity:**
- Review buildspec.yml
- Discuss pipeline stages
- Plan team's CI/CD strategy

## Architecture Review

**Duration: 2-4 hours**

### Topics:

**End-to-End ML Pipeline**
- Data ingestion
- Feature engineering
- Model training
- Deployment
- Monitoring
- Retraining

**High-Level Architecture Proposals**
- Single model deployment
- Multi-model deployment
- Microservices architecture
- Cost optimization strategies

**Group Exercise:**
- Design architecture for a use case
- Present to class
- Peer review and feedback

## Assessment & Evaluation

### Continuous Assessment
- Exercise completion
- Code quality
- Participation in discussions

### Final Evaluation
- Complete all four modules
- Deploy at least one model to SageMaker
- Implement monitoring
- Participate in architecture review

### Success Criteria
Students should be able to:
- [ ] Build and train ML models for various use cases
- [ ] Deploy models using appropriate strategies
- [ ] Set up monitoring and alerting
- [ ] Design MLOps pipelines
- [ ] Apply architectural best practices

## Common Student Questions

### Technical Questions

**Q: Why is my endpoint taking so long to create?**
A: Endpoint creation typically takes 5-10 minutes. This is normal as AWS provisions compute resources.

**Q: How do I reduce inference costs?**
A: Use auto-scaling, consider batch transform for non-urgent predictions, use multi-model endpoints when applicable.

**Q: My model training failed. What should I check?**
A: Check CloudWatch logs, verify IAM permissions, ensure S3 bucket access, check instance limits.

**Q: How often should I retrain models?**
A: Depends on data drift. Set up monitoring and retrain when performance degrades or on a schedule (weekly/monthly).

### Conceptual Questions

**Q: When should I use real-time vs batch inference?**
A: Real-time for latency-sensitive apps (<100ms), batch for periodic bulk predictions with flexible timing.

**Q: How do I choose between deployment strategies?**
A: Consider latency requirements, payload size, traffic patterns, and cost constraints.

**Q: What's the difference between SageMaker and training on EC2?**
A: SageMaker provides managed infrastructure, automatic scaling, built-in algorithms, and integrated monitoring.

## Troubleshooting Guide

### Setup Issues
1. **AWS Credentials**: Ensure proper IAM permissions
2. **Region Issues**: Verify region supports SageMaker
3. **Package Conflicts**: Use virtual environment

### Training Issues
1. **Out of Memory**: Use larger instances or reduce batch size
2. **Slow Training**: Use GPU instances for deep learning
3. **Convergence Issues**: Adjust learning rate, check data quality

### Deployment Issues
1. **Endpoint Creation Fails**: Check CloudWatch logs, verify permissions
2. **Inference Errors**: Validate input format, check model artifacts
3. **High Latency**: Optimize model, use auto-scaling, consider caching

## Best Practices for Instruction

### Do's ✓
- Encourage experimentation
- Share real-world experiences
- Facilitate peer learning
- Provide context for concepts
- Monitor AWS costs actively
- Be available during exercises
- Celebrate successes

### Don'ts ✗
- Don't rush through exercises
- Don't skip discussion time
- Don't ignore cost considerations
- Don't overcomplicate examples
- Don't forget to clean up resources

## Resource Management

### AWS Resources to Monitor
- SageMaker endpoints (most expensive)
- Training jobs
- Storage (S3, EBS)
- CloudWatch logs

### Daily Cleanup Checklist
- [ ] Delete unused endpoints
- [ ] Stop inactive notebook instances
- [ ] Remove old S3 objects
- [ ] Check CloudWatch logs retention

## Preparation Checklist

### Before Training Starts
- [ ] Test all notebooks in advance
- [ ] Verify AWS account access for all students
- [ ] Set up billing alerts
- [ ] Prepare backup exercises
- [ ] Review latest SageMaker features
- [ ] Test all deployment scripts
- [ ] Prepare discussion questions
- [ ] Set up communication channels

### During Training
- [ ] Monitor student progress
- [ ] Track AWS costs
- [ ] Collect feedback
- [ ] Adjust pace as needed
- [ ] Document common issues
- [ ] Facilitate discussions

### After Training
- [ ] Clean up all AWS resources
- [ ] Collect final feedback
- [ ] Share additional resources
- [ ] Provide completion certificates
- [ ] Update materials based on feedback

## Additional Resources

### For Instructors
- AWS SageMaker Developer Guide
- MLOps best practices papers
- Industry case studies
- Community forums

### For Students
- SageMaker examples repository
- AWS certification paths
- Machine learning papers
- Online communities

## Contact & Support

For instructor support:
- Training coordination team
- AWS technical support
- Instructor community forum
- Materials update requests

---

**Good luck with your training session!**
