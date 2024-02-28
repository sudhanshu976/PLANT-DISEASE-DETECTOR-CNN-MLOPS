from flask import Flask, render_template, request,jsonify
import tensorflow as tf
import subprocess
import numpy as np
from PIL import Image
import base64
import os
from PIL import Image
from numpy import asarray



app = Flask(__name__)
model = tf.keras.models.load_model('artifacts/saved_models/model.h5')



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file']

    image = Image.open(image)

    resized = image.resize((256, 256))
    data = asarray(resized)

    img_array = tf.expand_dims(data, 0)

    predictions = model.predict(img_array)

    result = np.argmax(predictions[0])

    if result == 0:
        result = 'This potato has a Early Blight Disease '
    elif result == 1:
        result = 'This potato has a Late Blight Disease '
    else:
        result = "This potato is Healthy"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)