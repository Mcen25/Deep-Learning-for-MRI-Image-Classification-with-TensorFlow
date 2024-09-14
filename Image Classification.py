import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

train_images, train_labels = next(iter(train_dataset))

# Code for plotting
def show(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.xlabel(class_names[labels[i]])
    plt.show()

show(train_images, train_labels)