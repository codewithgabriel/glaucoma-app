from flask import Flask, request, jsonify , render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = load_model('model/custom-glaucoma-model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']

    # Save the image temporarily
    img_path = os.path.join('temp', img.filename)
    img.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).resize((32, 32))  # Adjust size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image
    img_array = img_array / 255.0

    # Predict using the model
    prediction = model.predict(img_array)
    # print(prediction)
    predicted_class = np.argmax(prediction, axis=1)

    # Clean up the temporary image
    os.remove(img_path)
    print(prediction)
    
    prediction = prediction[0]


        
    result = {
        'nrg' : str(prediction[0]),
        'rg': str(prediction[1]),
    }
    print(result)
    # Return the prediction
    return jsonify(result)
if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(debug=True)
