from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
import io
import os

app = Flask(__name__)

# تحميل الموديل
model = load_model('Skin Cancer8.keras')

# التصنيفات الأصلية
original_labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
                   'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

# التصنيفات الجديدة (2 فقط)
new_labels = ['Benign', 'Malignant']

# خريطة التحويل
def map_to_new_label(original_index):
    malignant_classes = [0, 1, 5]  # المؤشرات الخاصة بـ Malignant
    if original_index in malignant_classes:
        return 'Malignant'
    else:
        return 'Benign'

def prepare_image(img_url, target_size=(224, 224)):
    response = requests.get(img_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
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
        original_label = original_labels[predicted_index]
        new_label = map_to_new_label(predicted_index)
        confidence = float(predictions[0][predicted_index])

        return jsonify({
            'original_label': original_label,
            'predicted_label': new_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
