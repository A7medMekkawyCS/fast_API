from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
import io

app = Flask(__name__)

# تحميل الموديل
model = load_model('Skin Cancer8.keras')

# أسماء الفئات (غيرها لو عندك أسماء تانية بالكود الأصلي)
labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
          'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# دالة لتحضير الصورة
def prepare_image(img_url, target_size=(224, 224)):  # عدل الحجم لو مختلف
    response = requests.get(img_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # تأكد لو الموديل متدرب normalized
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'imageUrl' not in data:
        return jsonify({'error': 'Missing imageUrl'}), 400
    
    img_url = data['imageUrl']
    
    try:
        img = prepare_image(img_url)
        predictions = model.predict(img)
        
        predicted_index = np.argmax(predictions[0])
        predicted_label = labels[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
