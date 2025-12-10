#!/usr/bin/env python
"""
Custom inference script for SageMaker container

Handles model loading, preprocessing, and predictions.
"""

import os
import json
import joblib
import flask
import pandas as pd
import numpy as np
from typing import Any, Dict

# Prefix for model artifacts
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# Global objects to be lazy-loaded
model = None
vectorizer = None
label_encoder = None


class ModelHandler:
    """Handles model loading and inference"""
    
    def __init__(self):
        """Initialize handler"""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
    
    def load_model(self):
        """Load model artifacts"""
        if self.model is None:
            self.model = joblib.load(os.path.join(model_path, "model.pkl"))
            
            # Load optional artifacts if they exist
            vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
            
            label_path = os.path.join(model_path, "label_encoder.pkl")
            if os.path.exists(label_path):
                self.label_encoder = joblib.load(label_path)
        
        return self.model
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions - works with both numeric and text data"""
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess based on data type
        if self.vectorizer is not None:
            # Text classification mode
            if 'text' in data.columns:
                X = self.vectorizer.transform(data['text'])
            elif 'review' in data.columns:
                X = self.vectorizer.transform(data['review'])
            else:
                # Assume all columns are text to vectorize
                text_data = data.iloc[:, 0].astype(str)
                X = self.vectorizer.transform(text_data)
        else:
            # Numeric/tabular mode - use data as-is
            X = data.values if hasattr(data, 'values') else data
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        # Decode labels if encoder exists
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        # Format response
        results = []
        for idx, pred in enumerate(predictions):
            result = {'prediction': int(pred) if isinstance(pred, (np.integer, int)) else str(pred)}
            
            if probabilities is not None:
                if self.label_encoder is not None:
                    result['probabilities'] = {
                        str(cls): float(prob) 
                        for cls, prob in zip(self.label_encoder.classes_, probabilities[idx])
                    }
                else:
                    result['probabilities'] = probabilities[idx].tolist()
                    result['confidence'] = float(max(probabilities[idx]))
            
            results.append(result)
        
        return {'predictions': results}


# Initialize handler
handler = ModelHandler()

# Flask app
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint"""
    try:
        handler.load_model()
        status = 200
    except Exception as e:
        status = 500
        print(f"Error loading model: {e}")
    
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """Inference endpoint - accepts JSON, CSV, or numpy arrays"""
    try:
        # Parse input based on content type
        if flask.request.content_type == "application/json":
            data = flask.request.get_json()
            # Handle different JSON formats
            if isinstance(data, dict):
                if 'instances' in data:
                    # Format: {"instances": [[1,2,3], [4,5,6]]}
                    df = pd.DataFrame(data['instances'])
                elif 'inputs' in data:
                    # Format: {"inputs": [[1,2,3], [4,5,6]]}
                    df = pd.DataFrame(data['inputs'])
                else:
                    # Format: {"feature1": [1,2], "feature2": [3,4]}
                    df = pd.DataFrame(data)
            elif isinstance(data, list):
                # Format: [[1,2,3], [4,5,6]]
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        elif flask.request.content_type == "text/csv":
            import io
            df = pd.read_csv(io.StringIO(flask.request.data.decode("utf-8")))
        else:
            return flask.Response(
                response=f"Unsupported content type: {flask.request.content_type}", 
                status=415, 
                mimetype="text/plain"
            )
        
        # Predict
        result = handler.predict(df)
        
        return flask.Response(
            response=json.dumps(result), 
            status=200, 
            mimetype="application/json"
        )
    
    except Exception as e:
        return flask.Response(
            response=json.dumps({"error": str(e)}), 
            status=500, 
            mimetype="application/json"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
