### Aman Nair 
### Brain tumour classification project
### 29th Oct 2022

# Importing required libraries
# - 
import os
import warnings
import itertools
import cv2
import seaborn as sns
import pandas as pd
import numpy  as np
from PIL import Image
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

import tensorflow as tf
import tensorflow_addons as tfa
import visualkeras
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection   import train_test_split
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

# Parameters for the model
# -
epochs = 15
pic_size = 240
np.random.seed(42)
tf.random.set_seed(42)

# Retrieving 
# - 
folder_path = "/Users/amannair/Downloads/Brain tumour detection"
positive = os.listdir('/Users/amannair/Downloads/Brain tumour detection/yes')
negative = os.listdir('/Users/amannair/Downloads/Brain tumour detection/no')
scans = []
labels = []

# Cleaning the data(Resizing the images and storing them in labeled arrays)
# - 
for image_name in negative:
    image=cv2.imread(folder_path + '/no/' + image_name)
    image=Image.fromarray(image,'RGB')
    image=image.resize((240,240))
    scans.append(np.array(image))
    labels.append(0)
    
for image_name in positive:
    image=cv2.imread(folder_path + '/yes/' + image_name)
    image=Image.fromarray(image,'RGB')
    image=image.resize((240,240))
    scans.append(np.array(image))
    labels.append(0)
    
scans = np.array(scans)
labels = np.array(labels)
print(scans.shape, labels.shape)

# Plotting first 10 images in each array
# - 
def plot_diag(diag):
    plt.figure(figsize= (24,24))
    for i in range(1, 10, 1):
        plt.subplot(3,3,i)
        img = load_img(folder_path + "/" + diag + "/" + os.listdir(folder_path + "/" + diag)[i], target_size=(pic_size, pic_size))
        plt.imshow(img)   
    plt.show()

plot_diag('yes')
plot_diag('no')

# Splitting the data into training and testing sets
# - 
x_train, x_test, y_train, y_test = train_test_split(scans, labels, test_size=0.2, shuffle=True, random_state=40)
(x_valid , y_valid) = (x_test[:63], y_test[:63])

# Defining the parameters of the neural network
# - 
model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid",input_shape=(pic_size,pic_size,3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), 
                          bias_regularizer=regularizers.L2(1e-2),
                          activity_regularizer=regularizers.L2(1e-3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy',
             optimizer=tf.keras.optimizers.Adam(),
             metrics=['acc'])

# fittting the data into the model
# - 
model.fit(x_train,
         y_train,
         batch_size=128,
         epochs=150,
         validation_data=(x_valid, y_valid),)

score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])

labels =["Yes",  
        "No"]

y_hat = model.predict(x_test)

# Plot a 10 test images at random with their predicted labels and ground truth
figure = plt.figure(figsize=(15, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    ax.set_title("{} ({})".format(labels[predict_index], 
                                  labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
plt.show()


