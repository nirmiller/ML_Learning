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

train_data_dir = r"Dataset\processed_data\train"
img_height, img_width = (224, 224)
batch_size = 32
valid_data_dir = r"DataSet\processed_data\val"

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

test_generator = train_datagen.flow_from_directory( valid_data_dir,
                                                     target_size=(img_height, img_width),
                                                     batch_size=1,
                                                     class_mode='categorical',
                                                     subset='validation')




model = tf.keras.models.load_model("Model/RestNet_Epoch_3")


filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob = []
y_act = []
test_generator.reset()
for _ in range(nb_samples):
    X_test, Y_test = test_generator.next()
    y_prob.append(model.predict(X_test))
    y_act.append(Y_test)

predicted_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
actual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act]

out_df = pd.DataFrame(np.vstack([predicted_class, actual_class]).T, columns=['predicted_class', 'actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'], out_df['predicted_class'], rownames=['Actual'], colnames=['Predicted'])
print("load heatmap")

sb.heatmap(confusion_matrix, cmap= 'Blues', annot=True,fmt='d')
plt.show()

print('test accuarcy : {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))

