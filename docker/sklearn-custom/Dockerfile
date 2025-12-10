# Custom Inference Container for SageMaker
# Uses scikit-learn with additional preprocessing libraries

FROM python:3.9-slim

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference code
COPY inference.py /opt/ml/code/
COPY serve /opt/ml/code/

# Make serve script executable
RUN chmod +x /opt/ml/code/serve

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Expose port for inference
EXPOSE 8080

# Start serving
ENTRYPOINT ["python", "/opt/ml/code/serve"]
