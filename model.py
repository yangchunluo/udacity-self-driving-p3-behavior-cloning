import csv
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

DRIVE_DATA_PATH = './driving-data'
IMAGE_SHAPE = (160, 320, 3)
CORRECTION = 0.2
BATCH_SIZE = 128

def data_generator(samples, batch_size):
    num_samples = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        # Generate one batch upon an invocation
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            batch_images = []
            batch_angles = []
            for subdir, path, angle, flip in batch_samples:
                path = os.path.join(DRIVE_DATA_PATH, subdir, 'IMG', os.path.basename(path))
                image = cv2.imread(path)
                if flip:
                    image = cv2.flip(image, 1)
                    angle *= -1.0
                batch_images.append(image)
                batch_angles.append(angle)

            X = np.array(batch_images)
            y = np.array(batch_angles)
            yield sklearn.utils.shuffle(X, y)

def read_driving_log():
    samples = []
    # Read the entire driving log (not images) into memory.
    for subdir in [
        'track1-forward',
        'track1-backward',
        'track1-left-recovery',
        'track1-right-recovery',
#        'track2-forward'
    ]:
        with open(os.path.join(DRIVE_DATA_PATH, subdir, 'driving_log.csv')) as f:
            reader = csv.reader(f)
            for line in reader:
                angle = float(line[3])
                center = line[0]
                left = line[1]
                right = line[2]
                # sub directory, image path (when catpured), angle, whether to flip it (for augmentation)
                samples.append((subdir, center, angle, True))
                samples.append((subdir, center, angle, False))
                if 'recovery' not in subdir:
                    samples.append((subdir, left, angle + CORRECTION, False))
                    samples.append((subdir, right, angle - CORRECTION, False))
    # The samples will be shuffled later.
    return samples

def build_model():
    model = Sequential()
    # Input normalization
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMAGE_SHAPE))
    # Cropping out unrelated image portions
    model.add(Cropping2D(cropping=((75, 25), (0, 0))))
    # Convolutional layers
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.5))
    # Output layer: no activation
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model

def main():
    samples = read_driving_log()
    print("Total number of samples %d" % len(samples))

    train_samples, valid_samples = train_test_split(samples, test_size=0.2)
    print("Number of training samples %d" % len(train_samples))
    print("Number of validation samples %d" % len(valid_samples))
    train_generator = data_generator(train_samples, BATCH_SIZE)
    valid_generator = data_generator(valid_samples, BATCH_SIZE)

    model = build_model()
    
    # Fit the model.
    model.fit_generator(train_generator, len(train_samples), nb_epoch=10,
                        validation_data=valid_generator, nb_val_samples=len(valid_samples))

    model.save('model.h5')

if __name__ == '__main__': main()
