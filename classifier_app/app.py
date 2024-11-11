import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and class labels
model = tf.keras.models.load_model('restnet.h5')
class_labels = ['Benign', 'Benign Without Callback', 'Malignant']

def preprocess_image(img):
    # Resize and convert to RGB
    img = img.convert("RGB").resize((224, 224))  
    # Normalize and add batch dim
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        img = Image.open(file)
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        return jsonify({
            'label': class_labels[predicted_class_index],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
