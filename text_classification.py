import os
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow_addons as tfa
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout,Activation,Input,BatchNormalization,GlobalAveragePooling1D
from keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50V2, ResNet50
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation
from tensorflow.keras.applications import EfficientNetB0
import cv2
gpus = tf.config.experimental.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)
all_df= pd.read_csv('data/A_Z Handwritten Data.csv')
all_df.shape
x = all_df.loc[0][1:].values
x = x.reshape((28, 28))
plt.imshow(x, cmap='binary')
plt.show()

all_df.head()
all_df['0'].value_counts().sort_index()


# load data from tensorflow framework
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Stack train data and test data to form single array 
mnist_data = np.vstack([x_train, x_test])

# Horizontal stacking labels of train and test set
mnist_labels = np.hstack([y_train, y_test])

# Uniques and counts of train labels
unique_train, counts_train = np.unique(y_train, return_counts= True)
print(f"Value counts of y_train modalities: {counts_train}\n")

# Uniques and counts of test labels
unique_test, counts_test = np.unique(y_test, return_counts= True)
print(f"Value counts of y_test modalities: {counts_test}")



def load_az_dataset(datasetPath):
    # List for storing data
    data = []
  
    # List for storing labels
    labels = []
  
    for row in open('data/A_Z Handwritten Data.csv'): #Openfile and start reading each row
    #Split the row at every comma
        row = row.split(",")
    
        #row[0] contains label
        label = int(row[0])
    
        #Other all collumns contains pixel values make a saperate array for that
        image = np.array([int(x) for x in row[1:]], dtype="uint8")
        
        #Reshaping image to 28 x 28 pixels
        image = image.reshape((28, 28))
        
        #append image to data
        data.append(image)
        
        #append label to labels
        labels.append(label)
    
    #Converting data to numpy array of type float32
    data = np.array(data, dtype='float32')
  
    #Converting labels to type int
    labels = np.array(labels, dtype="int")
    
    return (data, labels)

az_data, az_labels = load_az_dataset("data/A_Z Handwritten Data.csv")
az_data.shape
print(az_data.shape)
az_labels.shape
mnist_data
az_data
az_labels
mnist_labels
z = np.hstack([az_labels, mnist_labels])
z.shape
e = az_labels
e
a= az_labels
a
# the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are not incorrectly labeled 

az_labels += 10

# stack the A-Z data and labels with the MNIST digits data and labels

data = np.vstack([az_data, mnist_data])
labels = np.hstack([az_labels, mnist_labels])

# Each image in the A-Z and MNIST digts datasets are 28x28 pixels;
# However, the architecture we're using is designed for 32x32 images,
# So we need to resize them to 32x32

data = [cv2.resize(image, (32, 32)) for image in data]
data = np.array(data, dtype="float32")

# add a channel dimension to every image in the dataset and scale the
# pixel intensities of the images from [0, 255] down to [0, 1]

data = np.expand_dims(data, axis=-1)
data /= 255.0
data.shape
labels
le = LabelBinarizer()
labels = le.fit_transform(labels)

counts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]
classWeight
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)
trainX.shape
from skimage import io
aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.05,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
fill_mode="nearest")
batch_size = 50
epochs = 1
enet = ResNet50(
        input_shape=(32, 32, 1),
        weights=None,
        include_top=False
    )

model = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(36, activation='softmax')
    ])


callbacks = [ModelCheckpoint(filepath='best_model.h5', save_weights_only = True,
                             monitor='val_accuracy' ,mode='max')]
early = EarlyStopping(monitor="val_accuracy", mode="max", patience=20, verbose = 1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    epochs = epochs, 
    validation_data=(testX, testY),
    class_weight=classWeight,
    verbose=1,
    callbacks=[callbacks, early])
model.save('handwritten.model')


