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
import os
import pandas as pd
import tensorflow as tf






#dimensions
img_height, img_width = (224, 224)
batch_size = 32

#pre-processing images
train_data_dir = r"DataSet\processed_data\train"
valid_data_dir = r"DataSet\processed_data\val"
test_data_dir = r"DataSet\processed_data\test"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=.2,
                                   zoom_range=.2,
                                   horizontal_flip=True,
                                   validation_split=.4)

train_generator = train_datagen.flow_from_directory( train_data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     subset='training')
valid_generator = train_datagen.flow_from_directory( valid_data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     subset='validation')



test_generator = train_datagen.flow_from_directory( valid_data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=1,
                                                     class_mode='categorical',
                                                     subset='validation')


def train():
    x, y = test_generator.next()
    print(x.shape)

    base_model = ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)

    # Number of classes V
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    # Finale layer
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=5)

    model.save(b'Model\RestNet_Epoch_5')

    # model = load_model(os.path.join(save_path, b'ResNet_50Flowers'))

    test_loss, test_acc = model.evaluate(test_generator, verbose=2)
    print('\nTest Accuarcy', test_acc)


train()
