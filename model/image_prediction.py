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
class_names = ['roses', 'suse']


@app.route('/ai-demo', methods=['GET'])
def ai_demo():
    return 'AI DEMO', 200


@app.route('/training_result', methods=['GET'])
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


@app.route('/prediction', methods=['POST'])
def image_prediction():
    data = request.get_json()
    # load the image
    image_file = tf.keras.utils.get_file("image", origin=data['image_url'],
            force_download=True)
    img = tf.keras.utils.load_img(
        image_file, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return ("This image most likely belongs to {} with a {:.2f} percent"
            " confidence.\n\n").format(
                    class_names[np.argmax(score)], 100 * np.max(score)), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
