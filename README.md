# **Behavioral Cloning** 

---

Yangchun Luo

Nov 9, 2017

This is the assignment for Udacity's Self-Driving Car Term 1 Project 3.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

Note that during training the images were read in BGR format (`cv2.imread`). The received the image was in RGB format at inference time, which needed to be converted using `cv2.cvtColor(src, cv2.COLOR_RGB2BGR)`.

#### 3. Submission code is usable and readable

The model.py file contains the code for loading training data, and training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths of 6, 16, and 32 (model.py lines 71-77).

The model also contains three fully connected layers with sizes of 120, 84, and 1 (model.py lines 78-83).

The model includes RELU activation function to introduce nonlinearity, except for the output layer (model.py lines 80 and 82).

Data is normalized in the model using a Keras lambda layer (model.py line 68). The input image is also cropped to area that best represent the road condition (model.py line 70).

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting (model.py lines 81 and 83), with dropout rate 0.5, each after a fully-connected layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 94-98). The split rate was 80%-20% for training and validation. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually (model.py line 87).

#### 4. Appropriate training data

Training data was collected by driving on the center of the road, as well as recovering from both sides of the road. When driving on center, left and right camera images were used as additional training data with an empirical correction value to the steering angle.

A couple lessons learned during collecting driving data:

- Use mouse to steer the vehicle for a finer-grained steering angles.
- Keep the speed between 10-15 mph so that the chance for oversteering and correction is small. It also matches the target speed in autonomous mode.
- Create multiple directories to store the data for different purposes. In this way, a bad training run can be easily separated out and not "pollute" the entire training set.

Training data was collected from both track 1 and track 2. For details about how I created the training data, see the later section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to map the features of a front camera image to the best steering angle of the vehicle.

My first step was to use a convolution neural network model similar to the LeNet (i.e. two layers of 5x5 convolution with depth 6 and 16, respectively), since stacking the convolution and max-pooling layers was to proven to effective to extract proper image features.

The first model had both low training and validation errors. It worked okay for a short distance when applied to autonomous driving on the track. It got off the road at a major curve later. Another point of confusion came from the following point after the bridge, where the right side seemed to be a legitimate road as well:

<img src="writeup-images/confused-fork.jpg" alt="" width=300/>

It suggested that I'd need more training data, a more powerful model, as well as ways to mitigate overfitting.

- To increase the modeling power of the network, I added the third layer of convolution and max-pooling, and set the depth to 32, to learn some even higher level representation of the road features.

- I employed two layers of dropouts, one after each fully-connected layer--an experience learned from the previous project. The keep rate is to set 0.5 since it is in middle of the network (i.e. not input signals). I increased the training epochs to 10 as a result.

At the end of the process, also in combination of more training data and data augmentation (see later section), the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py build_model function) consisted of a convolution neural network with the following layers and layer sizes:


| Layer         		|     Description	        | 
|:-----------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   | 
| Lambda (Normalization) | outputs 160x320x3   | 
| Cropping			| outputs 60x320x3
| Convolution 5x5   | 1x1 stride, valid padding, outputs 56x316x6, RELU|
| Max pooling	      	| 2x2 stride,  outputs 28x158x6 |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 24x154x16, RELU|
| Max pooling	      	| 2x2 stride,  outputs 12x77x16 |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 8x73x32, RELU|
| Max pooling	      	| 2x2 stride,  outputs 4x36x32 |
| Fully connected   | input 4608 (4x36x32), outputs 120, RELU |
| Dropout           | rate 0.5 |
| Fully connected   | input 120, outputs 84, RELU |
| Dropout           | rate 0.5 |
| Fully connected   | input 84, outputs 1 |


Here is a visualization of the architecture using the built-in plot function in Keras.

<img src="writeup-images/model.png" alt="" width=150 />

#### 3. Creation of the Training Set & Training Process

Training data was collected on track 1 in the following manners:

* **Center-forward**: driving in counterclockwise direction, keeping the vehicle at center of the road, for two laps.
* **Center-backward**: driving in clockwise direction, keeping the vehicle at center of the road, for one lap.
* **Left-recovery**: driving in counterclockwise direction, recovering from left side of the road, for a couple of locations.
* **Right-recovery**: driving in counterclockwise direction, recovering from right side of the road, for a couple of locations.

For **Center-forward** and **Center-backward** data, both left and right camera images were used with an empirical correction value (0.2) to the steering angle.

| <img src="writeup-images/left_2017_11_08_19_17_16_172.jpg" width=175> | <img src="writeup-images/center_2017_11_08_19_17_16_172.jpg" width=175> | <img src="writeup-images/right_2017_11_08_19_17_16_172.jpg" width=175>
|:-----------------:|:---------------------------------------------:|:---|
| Left, correction +0.2 | Center, no correction | Right, correction -0.2

While these side cameras can help the model learn how to steer away from either side of the road, I also explicitly collected driving data by manually doing the same thing (**Left-recovery** and **Right-recovery**). Only center camera images were used in these cases.

| <img src="writeup-images/left-recovery.jpg" width=230> | <img src="writeup-images/right-recovery.jpg" width=230> |
|:-----------------:|:---------------------------------------------:|
| Left recovery | Right recovery |


All the *center* camera images are augmented by flipping the image along the X axis and reversing the steering angle (multiplied by -1.0). This creates additional dataset for training and validation.

Training data was also collected track 2 by driving forward and backward. I did not explicitly collected recovery driving data on track 2 due to time limitation, though.

Overall, there were a total of 55848 samples (including augmentation). At each epoch, 20% of the samples were used for validation and the rest for model fitting. I tried a total of 10 epochs, which is evidenced by the diminishing decrease of validation cost. I used an ADAM optimizer so that manually training the learning rate wasn't necessary.

```
44678/44678 [==============================] - 98s - loss: 0.0507 - val_loss: 0.0399
Epoch 2/10
44678/44678 [==============================] - 96s - loss: 0.0401 - val_loss: 0.0341
Epoch 3/10
44678/44678 [==============================] - 96s - loss: 0.0357 - val_loss: 0.0312
Epoch 4/10
44678/44678 [==============================] - 96s - loss: 0.0331 - val_loss: 0.0305
Epoch 5/10
44678/44678 [==============================] - 96s - loss: 0.0307 - val_loss: 0.0282
Epoch 6/10
44678/44678 [==============================] - 96s - loss: 0.0292 - val_loss: 0.0281
Epoch 7/10
44678/44678 [==============================] - 96s - loss: 0.0273 - val_loss: 0.0286
Epoch 8/10
44678/44678 [==============================] - 95s - loss: 0.0257 - val_loss: 0.0315
Epoch 9/10
44678/44678 [==============================] - 95s - loss: 0.0248 - val_loss: 0.0259
Epoch 10/10
44678/44678 [==============================] - 96s - loss: 0.0233 - val_loss: 0.0263
```

#### 4. Video Submission

I have included three autonomous driving videos in the submission.

- video.mpg shows the result of driving counterclockwise on track 1, as required by the rubirc.
- video2.mpg shows the result of driving clockwise on track 1.
- video3.mpg shows the result of driving on track 2.

The vehicle is able to stays on track 1 driving either direction. For track 2, due to the limited driving data (not included explicit recovery), there were two spots where the vehicle went off track. When manually corrected, it could finish the entire track.