# example taken from https://www.tensorflow.org/tutorials/images/classification

import boto3
from datetime import datetime
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# download dataset
#dataset_url = "http://gyee-ai-demo.s3-website.us-east-2.amazonaws.com/training_data-1.tgz"
dataset_url = os.environ['DATASET_URL']
data_dir = tf.keras.utils.get_file('training_data.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))
print('found ', image_count, ' images')

#suse = list(data_dir.glob('suse/*'))
#print(suse)
#PIL.Image.open(str(suse[0]))

#batch_size = 32
batch_size = 10
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print('image classes: ', class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

print("------------------ Training -----------------\n")

model_version = os.environ['MODEL_VERSION']
epochs=10
history_log_filename = 'training-' + str(model_version) + '.log'
model_filename = 'trained_model-' + str(model_version) + '.keras'
csv_logger = keras.callbacks.CSVLogger(history_log_filename)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[csv_logger]
)

print("Saving trained model")
model.save(model_filename)

print("Uploading model to S3")
# upload trained model and history to S3 bucket
s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
s3_client.upload_file(history_log_filename, 'gyee-ai-demo', history_log_filename)
s3_client.upload_file(model_filename, 'gyee-ai-demo', model_filename)
print("Done!")
