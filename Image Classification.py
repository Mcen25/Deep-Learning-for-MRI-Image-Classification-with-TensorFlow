import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_data = "data\Training"
testing_data = "data\Testing"

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    training_data,
    image_size=(256, 256),
    batch_size=32
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    testing_data,
    image_size=(256, 256),
    batch_size=32
)

train_images, train_labels = next(iter(train_dataset))
test_images, test_labels = next(iter(test_dataset))

# Code for plotting
# def show(images, labels):
#     plt.figure(figsize=(10, 10))
#     for i in range(16):
#         plt.subplot(4, 4, i + 1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.xlabel(class_names[labels[i]])
#     plt.show()

# show(train_images, train_labels)

#Normalize
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="same", activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax')) 
print(model.summary())  # Prints the model summary

# Use categorical crossentropy loss
loss = losses.CategoricalCrossentropy(from_logits=False)
optim = optimizers.Adam(learning_rate=0.0001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

batch_size = 64
epochs = 10

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)

model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=2)