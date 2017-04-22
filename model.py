
# coding: utf-8

# In[1]:

import csv
import cv2
import sklearn
import numpy as np
from keras_tqdm import TQDMNotebookCallback
from log_progress import log_progress

import sys
sys.stdout.isatty()

lines = []
base_path = "D:\\Car Engine\\Success 1\\"

samples = []

# I decided not to use a generator as I was training on a machine with 48gb of
# ram and could easily store the entire data set in memory.

with open(base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]
        steering_center = float(line[3])
        samples.append((center_path, steering_center, False))

        i+=1
        if i % 1000 == 0:
            print(i)

#Load the image
def load_image(sample):
    name = base_path+'IMG\\'+sample[0].split('\\')[-1]
    img = cv2.imread(name)
    return img

#Load all images and labels in
X_train = np.array([load_image(sample) for sample in log_progress(samples)])
Y_train = np.array([sample[1] for sample in log_progress(samples)])

# Description of the input images
ch, row, col = 3, 160, 320
dropout_rate = 0.2

#Building the model up
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Cropping2D, Conv2D
from keras.layers.core import Dropout
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch)))

#Crop the data down to a managable range. Remove the top 70 pixels, and the bottom 25.
model.add(Cropping2D(cropping=((70,25),(0,0))))

#Standard Nvidia driving architecture from lectures
model.add(Conv2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Conv2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Conv2D(48,5,5, subsample=(2,2), activation="relu"))

model.add(Conv2D(48,3,3, activation="relu"))
model.add(Conv2D(48,3,3, activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(dropout_rate))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print("Fitting")
import pdb; pdb.set_trace()
model.fit(X_train, Y_train, validation_split=0.2, verbose=0, epochs=10, shuffle=True)

print("Saving")
model_name = './model_temp.h5'
model.save(model_name)
print("Saved: ", model_name)
