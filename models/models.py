"""
# Python 3.5

This file contains all the architectures considered for this project.

By Shoaib Khan - Spring 2018
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LSTM
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras import regularizers
import configparser
K.set_image_dim_ordering('tf')

config = configparser.ConfigParser()
config.read('../config_nn/config_nn.ini')


def cnn_model(num_classes, shape):
    """
    This is a convolutional neural network
    :param num_classes:
    :param shape:
    :return:
    """
    model = Sequential()

    model.add(Conv2D(20, (3, 3), input_shape=shape, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(30, (3, 3), padding='same', strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.5))

    model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model


def simple_rnn_model(num_classes, shape):
    """
    This is a simple LSTM network
    :param num_classes:
    :param shape:
    :return:
    """
    model = Sequential()
    model.add(LSTM(10, input_shape=(None, shape), return_sequences=True))

    model.add(LSTM(100))
    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def logistic_regression(num_classes, dim):
    """
    This is a Logistic regression classifier
    :param num_classes:
    :param dim:
    :return:
    """
    output_dim = num_classes
    model = Sequential()
    model.add(Dense(output_dim, input_dim=dim, kernel_initializer='normal', activation='softmax'))
    return model


def neural_network(num_classes, dim):
    """
    This is a fully connected neural network
    :param num_classes:
    :param dim:
    :return:
    """
    model = Sequential()

    model.add(Dense(500, input_dim=dim, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    return model
