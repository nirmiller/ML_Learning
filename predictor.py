from tensorflow.keras.layers import Conv2D,Flatten, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import os
from tensorflow import keras

model = tf.keras.models.load_model("Model/RestNet_Epoch_1")

img_height, img_width = (224, 224)
batch_size = 32

name = "metalInternet.jpg"

flower_path = "testimages/{}.".format(name)


class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

img = keras.preprocessing.image.load_img(
    flower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(score)

index = np.argmax(score)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[index], 100 * np.max(score))
)
#Stuff

if(index == class_names.__len__()):
    print("Trash")
else:
    print("Recycle")