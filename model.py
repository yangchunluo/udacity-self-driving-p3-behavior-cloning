import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

DRIVE_DATA_PATH = './yc-drive-data'
IMAGE_SHAPE = (160, 320, 3)

def process_driving_log(data_root):
    images = []
    angles = []
    with open(os.path.join(data_root, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            # Correct the path which is generated on a different machine
            center_path = os.path.join(data_root, 'IMG', os.path.basename(line[0]))
            images.append(cv2.imread(center_path))
            angles.append(float(line[3]))
    return np.array(images), np.array(angles)

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=IMAGE_SHAPE))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_model(model, inputs, label):
    model.fit(inputs, label, validation_split=0.2, shuffle=True, nb_epoch=7)

def main():
    X_train, y_train = process_driving_log(DRIVE_DATA_PATH)
    model = build_model()
    train_model(model, X_train, y_train)
    model.save('model.h5')

if __name__ == '__main__': main()
