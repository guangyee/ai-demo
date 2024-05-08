import base64
import csv
from io import BytesIO
import matplotlib.pyplot as plt
import os
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras

from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api

app = Flask(__name__)

# TODO: make these configurable
model_version = os.environ['MODEL_VERSION']
training_log_url = 'http://gyee-ai-demo.s3-website.us-east-2.amazonaws.com/training-' + str(model_version) + '.log'
model_url = 'http://gyee-ai-demo.s3-website.us-east-2.amazonaws.com/trained_model-' + str(model_version) + '.keras'
model_file = tf.keras.utils.get_file("trained_model.keras", origin=model_url)
model = keras.models.load_model(model_file)
img_height = 180
img_width = 180
class_names = ['Chemeleon', 'SuSE Logo']


@app.route('/env', methods=['GET'])
def ai_demo_env():
    return "<h1><font color=\"blue\">Staging</font></h1>"


@app.route('/model-version', methods=['GET'])
def ai_demo():
    return "<h1>Model Version: <font color=\"red\">" + str(model_version) + "</font></h1>"


@app.route('/training-result', methods=['GET'])
def training_result():
    training_log_file = tf.keras.utils.get_file("training_log", origin=training_log_url)
    with open(training_log_file) as training_log:
        log = csv.DictReader(training_log)
        epochs = []
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        for row in log:
            epochs.append(row['epoch'])
            acc.append(row['accuracy'])
            val_acc.append(row['val_accuracy'])
            loss.append(row['loss'])
            val_loss.append(row['val_loss'])
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        return render_template('result.html', plot_url=plot_url)


@app.route('/prediction', methods=['GET', 'POST'])
def image_prediction():
    if request.method == 'GET':
        return render_template('image_upload.html')

    # load the image
    uploaded_file = request.files['file']
    uploaded_file.save(uploaded_file.filename)
    image_file = tf.keras.utils.get_file("image", origin="file:///work/" + uploaded_file.filename,
            force_download=True)
    img = tf.keras.utils.load_img(
        image_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return render_template('image_prediction.html',
                           image_file=os.path.join('work', uploaded_file.filename),
                           image_class=class_names[np.argmax(score)],
                           confidence="{:.2f}".format(100 * np.max(score)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
