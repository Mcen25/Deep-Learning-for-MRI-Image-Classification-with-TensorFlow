import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_data = "data/Training"
testing_data = "data/Testing"

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Data preprocessing
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    training_data,
    image_size=(256, 256),
    batch_size=64,
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    testing_data,
    image_size=(256, 256),
    batch_size=64,
)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_datagen = datagen
test_datagen = datagen

train_generator = train_datagen.flow_from_directory(
    training_data,
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    training_data,
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    testing_data,
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical'
)

# train_images, train_labels = next(iter(train_dataset))
# test_images, test_labels = next(iter(test_dataset))

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

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
print(model.summary())

# Use categorical crossentropy loss
loss = losses.CategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer='adam', loss=loss, metrics=metrics)

batch_size = 64
epochs = 10

# Convert labels to categorical
# train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)
# test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)

# model.fit(train_generator, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
)
# model.evaluate(test_generator, test_labels, batch_size=batch_size, verbose=2)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy:.2f}')