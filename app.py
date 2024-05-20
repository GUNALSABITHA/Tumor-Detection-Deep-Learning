from keras.preprocessing import image
import tflite
from keras.models import load_model
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

import tensorflow as tf
import keras
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model = load_model("models/Final-Model-FineTuned.keras")
model = tf.lite.Interpreter(model_path="models/converted_model.tflite")
model.allocate_tensors()
# load the tflite model using tflite package
# Load the TFLite model and allocate tensors.


app = Flask(__name__, template_folder='templates')


# def predict_class(model, path):
#     """
#     Predicts the class of the image at the given path
#
#     Returns an integer corresponding to the class of the image.
#     :param path: Path to the image to predict.
#     :param model: Model to use for prediction, must be a Keras model (load_model) object.
#     :type path:
#     :return: int
#     :rtype:
#     """
#     img = Image.open(path)
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     if len(img_array.shape) == 2:
#         img_array = img_array.reshape((224, 224, 1))
#     # Convert from uint8 (0-255) to float32 (0.0-1.0) for the model
#     img_array = img_array.astype('float32') / 255.0
#
#     threshold = 0.5
#     prediction = model.predict(img_array[np.newaxis])  # Make prediction
#     classification = 1 if prediction[0][0] > threshold else 0  # Apply threshold
#
#     # Print the classification result (0 - Brain Tumor, 1 - Healthy)
#     return classification

def predict_class(model, path):
    """
    Predicts the class of the image at the given path

    Returns an integer corresponding to the class of the image.
    :param path: Path to the image to predict.
    :param model: Model to use for prediction, must be a TensorFlow Lite Interpreter object.
    :type path:
    :return: int
    :rtype:
    """
    img = Image.open(path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = img_array.reshape((224, 224, 1))
    # Convert from uint8 (0-255) to float32 (0.0-1.0) for the model
    img_array = img_array.astype('float32') / 255.0

    # Set the tensor to point to the input data to be inferred
    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], img_array[np.newaxis])

    # Run the inference
    model.invoke()

    # Get the output of the inference
    output_details = model.get_output_details()
    output_data = model.get_tensor(output_details[0]['index'])

    threshold = 0.5
    classification = 1 if output_data[0][0] > threshold else 0  # Apply threshold

    # Print the classification result (0 - Brain Tumor, 1 - Healthy)
    return classification


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    file = request.files.get('image')

    if not file:
        return "No file uploaded", 400

    #     display the image in the server backend
    img = Image.open(file)

    # Save the image to the server
    img_path = os.path.join("uploads", secure_filename(file.filename))
    img.save(img_path)

    # Predict the class of the image
    prediction = predict_class(model=model, path=img_path)

    print(prediction * 100)

    if prediction == 0:
        return redirect(url_for("tumor"))
    else:
        return redirect(url_for("healthy"))


@app.route("/tumor")
def tumor():
    return render_template("tumor.html")


@app.route("/healthy")
def healthy():
    return render_template("healthy.html")


if __name__ == "__main__":
    app.run(debug=True)
