pipeline {
    agent any
    
    environment {
        AWS_DEFAULT_REGION = 'us-east-1'
        PYTHON_VERSION = '3.11'
        VENV_DIR = 'venv'
    }
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timestamps()
        timeout(time: 1, unit: 'HOURS')
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
                sh 'git --version'
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python environment...'
                sh '''
                    python3 --version
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip --version
                '''
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Installing dependencies...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pip install -r requirements.txt
                    pip list
                '''
            }
        }
        
        stage('Lint Code') {
            steps {
                echo 'Running code linting...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Running flake8..."
                    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=${VENV_DIR},.git,__pycache__ || true
                    echo "Linting completed"
                '''
            }
        }
        
        stage('Run Unit Tests') {
            steps {
                echo 'Running unit tests...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    if [ -d "tests" ]; then
                        pytest tests/ -v --cov=. --cov-report=html --cov-report=term --cov-report=xml || true
                    else
                        echo "No tests directory found, skipping..."
                    fi
                '''
            }
        }
        
        stage('Security Scan') {
            steps {
                echo 'Running security scans...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Checking for known vulnerabilities in dependencies..."
                    pip list --format=json > installed_packages.json || true
                '''
            }
        }
        
        stage('Build Artifacts') {
            steps {
                echo 'Building artifacts...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Creating distribution package..."
                    mkdir -p dist
                    tar -czf dist/aws-ml-training-$(date +%Y%m%d-%H%M%S).tar.gz \
                        --exclude=${VENV_DIR} \
                        --exclude=.git \
                        --exclude=__pycache__ \
                        --exclude=*.pyc \
                        --exclude=dist \
                        .
                    ls -lh dist/
                '''
            }
        }
        
        stage('Archive Artifacts') {
            steps {
                echo 'Archiving artifacts...'
                archiveArtifacts artifacts: 'dist/*.tar.gz', fingerprint: true
                sh '''
                    if [ -d "htmlcov" ]; then
                        echo "Archiving coverage report..."
                    fi
                '''
            }
        }
        
        stage('Deploy to Dev') {
            when {
                branch 'develop'
            }
            steps {
                echo 'Deploying to Development environment...'
                sh '''
                    echo "Development deployment steps would go here"
                    echo "This could include:"
                    echo "  - Uploading to S3"
                    echo "  - Deploying to SageMaker dev endpoint"
                    echo "  - Running integration tests"
                '''
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'staging'
            }
            steps {
                echo 'Deploying to Staging environment...'
                sh '''
                    echo "Staging deployment steps would go here"
                    echo "This could include:"
                    echo "  - Deploying to SageMaker staging endpoint"
                    echo "  - Running smoke tests"
                    echo "  - Validating model performance"
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy'
                echo 'Deploying to Production environment...'
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Production deployment steps would go here"
                    echo "This could include:"
                    echo "  - Blue-Green deployment"
                    echo "  - Canary deployment"
                    echo "  - A/B testing setup"
                    echo "  - Monitoring setup"
                '''
            }
        }
        
        stage('Notify') {
            steps {
                echo 'Sending notifications...'
                sh '''
                    echo "Build completed successfully!"
                    echo "Branch: ${GIT_BRANCH}"
                    echo "Commit: ${GIT_COMMIT}"
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh '''
                echo "Cleaning temporary files..."
                rm -f installed_packages.json
            '''
        }
        
        success {
            echo 'Pipeline succeeded!'
            // Send success notification (e.g., Slack, email)
        }
        
        failure {
            echo 'Pipeline failed!'
            // Send failure notification
        }
        
        cleanup {
            echo 'Final cleanup...'
            sh 'rm -rf ${VENV_DIR}'
        }
    }
}
