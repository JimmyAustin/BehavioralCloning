# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_driving_image]: ./examples/center_driving.jpg "Center Driving"
[recovery_1]: ./examples/recovery_1.jpg "Oh no"
[recovery_2]: ./examples/recovery_2.jpg "Oh no"
[recovery_3]: ./examples/recovery_3.jpg "Oh no"
[input]: ./examples/input.png "Input"
[output]: ./examples/output.png "Output"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with filter size 5x5 and depths between 24 and 48. It is a reimplementation of the Nvidia architecture used in the lectures. It uses RELU layers to add nonlinearity (lines 61-66), and normalizes the data (line 55).

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model includes dropout layers to avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle in the middle of the road. Only the center camera was used. All training was done on track 1.

The training set consisted of:
- 3 clockwise laps
- 3 anticlockwise laps
- 5 clockwise trips across the bridge
- 5 clockwise trips across the bridge
- 2 clockwise laps whilst swerving over both sides of the road. Only corrections were recorded.
- 2 anticlockwise laps whilst swerving over both sides of the road. Only corrections were recorded.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the suggested nvidia architecture.

Initially this was a failure, as the custom training set was not comprehensive enough. As the first iteration consisted of only going around the track in one direction, the training set had a statistical leaning towards turning in one direction, causing it to go in circles.

Upon augmenting the data with laps in both directions, it would successfully drive forwards, but couldnt turn properly.

I then trained the car to correct by recording recovering from swerving on both sides of the road. This seemed to give the car the ability to self correct if it took a corner to sharply.

This proved to be largely successful, bar some strange behavior on the bridge. This was remedied by doing an additional 10 laps in each direction.

Overfitting was largely avoided by not training for very long. Scientifically not very accurate, but I was training on a 7 year old desktop and its not exactly a powerhouse.

####2. Final Model Architecture

The final model architecture consisted on the following convolutional neural network

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input | 160x320x3 |
| Normalisation         		| Moving the data to be between -0.5 and 0.5|
| Cropping | Removes top 70 pixels and bottom 25 pixels. Outputs 64x320x3 |
| Convolution 5x5 | 2x2 stride, valid padding, outputs 31x158x24 |
| RELU |  |
| Convolution 5x5 | 2x2 stride, valid padding, outputs 14x77x36 |
| RELU |  |
| Convolution 5x5 | 2x2 stride, valid padding, outputs 5x37x48 |
| RELU |  |
| Convolution 3x3 | 1x1 stride, outputs 3x35x48 |
| RELU |  |
| Convolution 3x3 | 1x1 stride, outputs 1x33x48 |
| RELU |  |
| Flatten | Outputs 1584|
| Fully Connected | Outputs 100 |
| Dropout | 20% |
| Fully Connected | Outputs 50 |
| Dropout | 20% |
| Fully Connected | Outputs 10 |
| Dropout | 20% |
| Fully Connected | Outputs 1 |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_driving_image]

Unfortunately this dataset was not enough to adequately train the model, and the car simply drove in circles. This is potentially because every clockwise lap will have a leaning towards turning right.

I added to the dataset by doing 5 laps in the other direction. This caused the car to drive straight until it hit a corner where it would turn a little, then decide to go out on its own in the world and explore the bottom of the lake. Obviously this is unintended functionality.

I then trained the car to correct by recording recovering from swerving on both sides of the road. This seemed to give the car the ability to self correct if it took a corner to sharply.

![alt text][recovery_1]
Nearing the edge of the road

![alt text][recovery_2]
Correcting

![alt text][recovery_3]
Corrected

The dataset was not augmented, and left and right camera outputs were discarded.

All up, the final model was trained on 8491 datapoints.

It was then preprocessed by cropping the top 70 pixels and the bottom 25px. The dark band in the following picture is what remained.

![alt text][input]
![alt text][output]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The 20% validation set helped determine the success of the model. I trained it for 10 epochs. validation loss was still falling at the end, but given that the model was successful in achieving the project's goals and it was late, I decided that was enough.

An adam optimizer was used, so no learning rate was set.
