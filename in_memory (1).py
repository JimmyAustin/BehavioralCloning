from log_progress import log_progress

import csv
import cv2
import sklearn
import numpy as np
from keras_tqdm import TQDMNotebookCallback

lines = []
base_path = "D:\\Car Engine\\Track 1\\"

samples = []

correction = 0.2

with open(base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for line in reader:
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]
        steering_center = float(line[3])
        samples.append((center_path, steering_center))
        #samples.append((left_path, steering_center + correction))
        #samples.append((right_path, steering_center - correction))
        i+=1
        if i % 1000 == 0:
            print(i)


# In[4]:

def process_image(x):
    return x/127.5 - 1.

def load_image(sample):
    name = base_path+'IMG\\'+sample[0].split('\\')[-1]
    img = cv2.imread(name)
    return img
#    return process_image(cv2.imread(name))

print("Loading in dataset")
X_train = np.array([load_image(sample) for sample in log_progress(samples)])
print("Loaded")
Y_train = np.array([sample[1] for sample in log_progress(samples)])


# In[5]:

ch, row, col = 3, 160, 320


# In[6]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
          input_shape=(row, col, ch),
          output_shape=(row, col, ch)))
#model.add(Dense(32, input_shape=(row, col, ch)))


model.add(Flatten())
model.add(Dense(1))


print("Model built.")
# In[7]:

print("Model Compiling.")
model.compile(loss='mse', optimizer='adam')
print("Model compiled")
if True:
    print("Fitting, non terminal")
    model.fit(X_train, Y_train, epochs=3, shuffle=True)
else:
    model.fit(X_train, Y_train, verbose=0, callbacks=[TQDMNotebookCallback()], epochs=3)

print("Saving")
model_name = './model_2.h5'
model.save(model_name)
print("Saved: ", model_name)
