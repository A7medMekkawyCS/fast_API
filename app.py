from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
import io
import os
<<<<<<< HEAD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
=======

>>>>>>> 04b225943634cd110118e843e2ed95e607d266e2
app = Flask(__name__)

# تحميل الموديل
model = load_model('Skin cancer11.keras')

# التصنيفات المتوقعة
<<<<<<< HEAD
labels = ['Benign','Malignant']
=======
labels = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
          'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
>>>>>>> 04b225943634cd110118e843e2ed95e607d266e2

# معالجة الصورة
def prepare_image(img_url, target_size=(224, 224)): 
    response = requests.get(img_url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
<<<<<<< HEAD
    img_array = preprocess_input(img_array)  # Normalize input for MobileNet
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension
=======
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
>>>>>>> 04b225943634cd110118e843e2ed95e607d266e2
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
        threshold = 0.965  # Increase the threshold to be more confident
        predicted_label = 'Malignant' if predictions[0] >= threshold else 'Benign'
        return jsonify({
            'predicted_label': predicted_label
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# إعداد المنفذ حسب Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
