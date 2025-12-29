"""
Diabetic Retinopathy Detection - Flask REST API
Author: Keerthi Samhitha Kadaveru
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import base64

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = 'src/Model/dr_model.h5'
IMG_SIZE = 512
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Load model
model = None

def load_dr_model():
    """Load the trained model"""
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Apply CLAHE for contrast enhancement
    img_lab = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB) / 255.0
    
    return np.expand_dims(img_enhanced, axis=0)

def predict_single_eye(image):
    """Make prediction for a single eye"""
    if model is None:
        return {"error": "Model not loaded"}
    
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])
    
    return {
        "class": CLASS_NAMES[predicted_class],
        "class_index": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in predictions]
    }

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        "message": "Diabetic Retinopathy Detection API",
        "endpoints": {
            "/predict": "POST - Upload left_eye and right_eye images",
            "/health": "GET - Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diabetic retinopathy from retinal images
    
    Expects:
        - left_eye: Image file
        - right_eye: Image file
    
    Returns:
        JSON with predictions for both eyes
    """
    try:
        # Check if files are present
        if 'left_eye' not in request.files or 'right_eye' not in request.files:
            return jsonify({
                "error": "Both left_eye and right_eye images are required"
            }), 400
        
        left_file = request.files['left_eye']
        right_file = request.files['right_eye']
        
        # Read images
        left_image = Image.open(left_file.stream)
        right_image = Image.open(right_file.stream)
        
        # Make predictions
        left_result = predict_single_eye(left_image)
        right_result = predict_single_eye(right_image)
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "left_eye": left_result,
            "right_eye": right_result,
            "summary": {
                "worse_eye": "left" if left_result["class_index"] > right_result["class_index"] else "right",
                "max_severity": CLASS_NAMES[max(left_result["class_index"], right_result["class_index"])]
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate PDF report for the diagnosis"""
    try:
        if 'left_eye' not in request.files or 'right_eye' not in request.files:
            return jsonify({"error": "Both images required"}), 400
        
        left_file = request.files['left_eye']
        right_file = request.files['right_eye']
        
        left_image = Image.open(left_file.stream)
        right_image = Image.open(right_file.stream)
        
        left_result = predict_single_eye(left_image)
        right_result = predict_single_eye(right_image)
        
        # Generate PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(180, height - 50, "Diabetic Retinopathy Report")
        
        # Date
        c.setFont("Helvetica", 12)
        c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Left Eye Results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 130, "Left Eye:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 150, f"Classification: {left_result['class']}")
        c.drawString(70, height - 170, f"Confidence: {left_result['confidence']*100:.1f}%")
        
        # Right Eye Results
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 210, "Right Eye:")
        c.setFont("Helvetica", 12)
        c.drawString(70, height - 230, f"Classification: {right_result['class']}")
        c.drawString(70, height - 250, f"Confidence: {right_result['confidence']*100:.1f}%")
        
        # Summary
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 300, "Summary:")
        c.setFont("Helvetica", 12)
        max_severity = CLASS_NAMES[max(left_result["class_index"], right_result["class_index"])]
        c.drawString(70, height - 320, f"Maximum Severity: {max_severity}")
        
        # Disclaimer
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, 50, "This is an AI-assisted screening tool. Please consult an ophthalmologist for clinical diagnosis.")
        
        c.save()
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'DR_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_dr_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
