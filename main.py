from detect import predict_class
import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/predict_Disease', methods=['POST'])
def detect_deficiency_api():
    image = request.files['image']
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    print(type(img))
    
    virus = predict_class(img)
    return str(virus)  # Convert the result to a string

@app.route('/')
def test_page():
    categories = 'categories.html'
    return render_template('index.html',categories = categories)

if __name__ == '__main__':
    app.run()
