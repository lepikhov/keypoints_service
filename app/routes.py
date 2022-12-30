from app import app
from app.processing import predict_result
from flask import Flask, jsonify, request


@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"       

   
    file = request.files.get('file')

    if not file:
        return "There is not file"

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    #img = prepare_image(img_bytes)
 
    # Return on a JSON format
    return jsonify(predict_result(img_bytes))
    


@app.route('/', methods=['GET'])
def index():
    return 'Service for automatic determination of the key points of the horse'
