# from keras.preprocessing import image
# import tflite
# from keras.models import load_model
from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image

import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter
# import keras
import numpy as np
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# model = load_model("models/Final-Model-FineTuned.keras")
model = Interpreter(model_path="models/converted_model.tflite")
# model = tf.Interpreter(model_path="models/converted_model.tflite")
model.allocate_tensors()
# load the tflite model using tflite package
# Load the TFLite model and allocate tensors.


app = Flask(__name__, template_folder='templates')


def predict_class(model, file):
    """
    Predicts the class of the image from the given file object

    Returns an integer corresponding to the class of the image.
    :param file: File object to predict.
    :param model: Model to use for prediction, must be a TensorFlow Lite Interpreter object.
    :type file: werkzeug.datastructures.FileStorage
    :return: int
    :rtype:
    """
    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = img_array.reshape((224, 224, 1))
    img_array = img_array.astype('float32') / 255.0

    input_details = model.get_input_details()
    model.set_tensor(input_details[0]['index'], img_array[np.newaxis])

    model.invoke()

    output_details = model.get_output_details()
    output_data = model.get_tensor(output_details[0]['index'])

    threshold = 0.5
    classification = 1 if output_data[0][0] > threshold else 0

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

    # Predict the class of the image
    prediction = predict_class(model=model, file=file)

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
