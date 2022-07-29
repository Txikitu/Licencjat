import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import keras
import tensorflow as tf
from tensorflow.keras import layers

import os
import splitfolders
import pathlib

seed = 21

num_classes = 6
img_height = 256
img_width = 256

for folder in os.listdir('Data'):
    splitfolders.ratio('Data/{}'.format(folder), output='output', seed=seed, ratio=(.7, 0.13, 0.17))

splitfolders.ratio('Data', output='output', seed=seed, ratio=(.7, 0.13, 0.17))

model = keras.applications.resnet50.ResNet50(weights="imagenet")

images = tf.keras.utils.get_file(origin='D:\SGH\Praca licencjacka\Kod\Data',
                                   fname='Data')
data_dir = pathlib.Path('D:\SGH\Praca licencjacka\Kod\Data')
image_count = len(list(data_dir.glob('*/*.jpg')))

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\SGH\Praca licencjacka\Kod\output\train',
    labels="inferred",
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.1566,
    subset='training',
    interpolation="bilinear",
    follow_links=False,
)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\SGH\Praca licencjacka\Kod\output\train',
    labels="inferred",
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    validation_split=0.1566,
    subset='validation',
    interpolation="bilinear",
    follow_links=False,
)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\SGH\Praca licencjacka\Kod\output\train',
    labels="inferred",
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    interpolation="bilinear",
    follow_links=False,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    r'D:\SGH\Praca licencjacka\Kod\output\val',
    labels="inferred",
    label_mode='int',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=123,
    interpolation="bilinear",
    follow_links=False,
)


import matplotlib.pyplot as plt

class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)


model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
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

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_drop_out = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model_drop_out.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 15
history = model_drop_out.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

