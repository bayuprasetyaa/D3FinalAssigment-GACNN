from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Helper: Early Stopper
early_stopper = EarlyStopping(patience=5)


def get_dataset():
    """
    Get and process the dataset
    :return:
    """
    # Input patch
    data_dir = "dataset/fer2013/fer2013.csv"

    # Ekstracting the image from csv
    x = []
    y = []
    first = True
    for line in open(data_dir):
        if first:
            first = False
        else:
            row = line.split(',')
            x.append([int(p) for p in row[1].split()])
            y.append(int(row[0]))
    x, y = np.array(x) / 255.0, np.array(y)

    x = x.reshape(-1, 48, 48, 1)

    # Spliting data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Set default
    batch_size = 64
    target_size = (50, 50)
    input_shape = (48, 48, 1)
    classes = len(set(y))

    y_train = (np.arange(classes) == y_train[:, None]).astype(np.float32)
    y_test = (np.arange(classes) == y_test[:, None]).astype(np.float32)

    # Adding image augmentation
    datagen = ImageDataGenerator(
        zoom_range=0.15,
        height_shift_range=0.15,
        width_shift_range=0.15,
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True)

    datagen.fit(x_train)

    """
    train = data_generator.flow_from_directory(
        data_dir,
        target_size,
        #classes=classes,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical'
    )

    validation = data_generator.flow_from_directory(
        data_dir,
        target_size,
        #classes=classes,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical'
    )
     """

    return input_shape, classes, x_train, x_test, y_train, y_test, datagen, batch_size


def compile_model(network, input_shape, classes):
    """
    Compile Sequential model

    :return: compiled model
    """
    # Get network parameter
    cov2d_layers = network['cov2d_layers']
    neurons = network['neurons']
    optimizer = network['optimizer']
    convolution = network['convolution']
    fc_layers = network['fc_layers']

    model = Sequential()

    counter = 0
    i = 0
    j = 0

    # Add convolutional layer
    for layer in cov2d_layers:
        # Need input shape for first layer.
        if i < convolution:
            if counter == 0:
                model.add(Conv2D(layer, (3, 3), input_shape=input_shape, activation='relu'))
                model.add(MaxPool2D(2, 2))
            else:
                model.add(Conv2D(layer, (3, 3), activation='relu'))
                model.add(MaxPool2D(2, 2))

            counter += 1
        else:
            break
        i += 1

    model.add(Flatten())

    # Add fully-connected layer
    for layer in neurons:
        if j < fc_layers:
            model.add(Dense(layer, activation='relu'))
        else:
            break
        j += 1

    # Add Output layer
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    return model


def train_and_score(network):
    """
    Train the model
    :param network: the parameter of network
    :return: Loss
    """

    input_shape, classes, x_train, x_test, y_train, y_test, datagen, batch_size = get_dataset()

    model = compile_model(network, input_shape, classes)

    model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
              epochs=15,
              validation_data=(x_test, y_test),
              verbose=1)

    """
    model.fit_generator(train,
                        epochs=15,
                        verbose=1,
                        validation_data=validation,
                        callbacks=[early_stopper])
    """

    score = model.evaluate(x_test, y_test)

    return score[1]
