import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

DRIVE_DATA_PATH = './yc-drive-data'
IMAGE_SHAPE = (160, 320, 3)
CORRECTION = 0.2

def process_driving_log(data_root):
    images = []
    angles = []
    # Load data from the simulation and correct the path.
    with open(os.path.join(data_root, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            # Process center, left, and right images
            # Center: CSV offset 0, no correction
            # Left:   CSV offset 1, correction to right (plus)
            # Right:  CSV offset 2, correction to left  (minus)
            camera_pos = line[0:3]
            correction = [0.0, CORRECTION, -CORRECTION]
            for f, c in zip(camera_pos[0:1], correction[0:1]):
                path = os.path.join(data_root, 'IMG', os.path.basename(f))
                images.append(cv2.imread(path))
                angles.append(float(line[3]) + c)

    # Augment the data by flipping it.
    flipped_images = []
    flipped_angles = []
    for image, angle in zip(images, angles):
        flipped_images.append(cv2.flip(image, 1))
        flipped_angles.append(angle * -1.0)

    return np.array(images + flipped_images), np.array(angles + flipped_angles)

def build_model():
    model = Sequential()
    # Input normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGE_SHAPE))
    print(model.layers[-1].output_shape)
    # Cropping out unrelated image portions
    model.add(Cropping2D(cropping=((75, 25), (0, 0))))
    print(model.layers[-1].output_shape)
    # Convolutional layers
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    print(model.layers[-1].output_shape)
    model.add(MaxPooling2D())
    print(model.layers[-1].output_shape)
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    print(model.layers[-1].output_shape)
    model.add(MaxPooling2D())
    print(model.layers[-1].output_shape)
    # Fully connected layers
    model.add(Flatten())
    print(model.layers[-1].output_shape)
    model.add(Dense(120, activation='relu'))
    print(model.layers[-1].output_shape)
    model.add(Dense(84, activation='relu'))
    print(model.layers[-1].output_shape)
    # Output layer: no activation
    model.add(Dense(1))
    print(model.layers[-1].output_shape)

    model.compile(loss='mse', optimizer='adam')
    return model

def train_model(model, inputs, label):
    model.fit(inputs, label, validation_split=0.2, shuffle=True, nb_epoch=5)

def main():
    X, y= process_driving_log(DRIVE_DATA_PATH)
    print("Total number of images %d" % len(X))

    model = build_model()
    train_model(model, X, y)
    model.save('model.h5')

if __name__ == '__main__': main()
