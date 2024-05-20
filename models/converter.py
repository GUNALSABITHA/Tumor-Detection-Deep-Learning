import tensorflow as tf

model = tf.keras.models.load_model("Final-Model-FineTuned.keras")

tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_converter.convert()

with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)
