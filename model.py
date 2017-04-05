
# coding: utf-8

# In[50]:

import csv
import cv2
import sklearn
import numpy as np


# In[82]:

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


# In[111]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = base_path+'IMG\\'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def in_memory_generator(samples, batch_size=32):
    print("Starting IM")
    def load_image(sample):
        name = base_path+'IMG\\'+sample[0].split('\\')[-1]
        return cv2.imread(name)
    images = np.array([load_image(sample) for sample in samples])
    angles = np.array([sample[1] for sample in samples])
    num_samples = len(samples)
    print("Loaded")
    while 1:
        for offset in range(0, num_samples, batch_size):
            X_train = images[offset:offset+batch_size]
            y_train = angles[offset:offset+batch_size]
            yield sklearn.utils.shuffle(X_train, y_train)
# In[112]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[113]:

# compile and train the model using the generator function
train_generator = in_memory_generator(train_samples, batch_size=32)
print("MURDER TRAIN")
validation_generator = in_memory_generator(validation_samples, batch_size=32)

print("train length: ", len(train_samples))
print("valid length: ", len(validation_samples))
# In[114]:

ch, row, col = 3, 160, 320


# In[117]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Flatten())
model.add(Dense(1))



# In[ ]:

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch= (int(len(train_samples)/32) - 1),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=3)


# In[ ]:
