"""
# Python 3.5

This file contains functions that will be used by the run-classifiers.py script

By Shoaib Khan - Spring 2018
"""

import pickle
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import models as m
from keras.metrics import top_k_categorical_accuracy
import h5py
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model


def combine1d(signal1, signal2, label_index):
    """
    Combines two signals into one output signal.
    """
    combined = []
    combined.extend(np.ravel(signal1))
    combined.extend(np.ravel(signal2))
    combined.append(label_index)
    return combined


def pad(signal, new_length, end):
    """
    Pads the input signals
    """
    assert len(signal) > 1 and len(signal[0]) > 1
    signal = np.array(signal)
    if len(signal) < new_length:
        zero_row = np.zeros(len(signal[0]))
        zero_row = np.array([zero_row])
        count = 0
        while len(signal) < new_length:
            if end:
                signal = np.concatenate((signal, zero_row))
            else:
                if count % 2 == 0:
                    signal = np.concatenate((zero_row, signal))
                else:
                    signal = np.concatenate((signal, zero_row))
            count += 1
    return signal[:new_length]


def pad_rows(signal_array, new_length, end):
    """
    Pads arrays of signals
    """
    new_array = []
    for signal in signal_array:
        new_array.append(pad(signal, new_length, end))
    return np.array(new_array)


def save_object(obj, destination):
    """
    Saves objects to a destination
    """
    print("Saving pickle object")
    with open(destination, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(source):
    """
    Loads objects from a source
    """
    print("Loading pickle object")
    with open(source, 'rb') as s:
        return pickle.load(s)


def create_directory(destination):
    """
    Creates directories
    """
    if not os.path.exists(destination):
        os.makedirs(destination)


def create_required_directories():
    """
    Creates required directories in the models directory
    """
    create_directory("checkpoints")
    create_directory("confusion_matrix")
    create_directory("logs")
    create_directory("model_plots")
    create_directory("normalizers")
    return


def read_files(file_path, subjects, folder_name):
    """
    Reads and parses input files
    """
    x_out = []
    paths = []
    for subject in subjects:
        path = file_path.format(folder_name, subject)
        paths.append(path)
        for file in glob.glob(path):
            with h5py.File(file, "r") as opened:
                combined = {}
                for k in opened.keys():
                    combined[k] = opened[k].value
                x_out.append([combined["signal"], combined["label"]])
    return x_out, paths


def split_x_y(x):
    """
    Split into input (X) and output (Y) variables
    """
    length = len(x[0])
    y_labeled = x[:, length - 1]
    x = x[:, 0:1]
    y = to_categorical(y_labeled)

    new_x = []
    for i in x:
        new_x.append(i[0])
    x = np.array(new_x)

    return x, y, y_labeled


def get_metrics():
    """
    Creates custom top-n metrics
    """
    def top2(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top3(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top4(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=4)

    def top5(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)
    return ['accuracy', top2, top3, top4, top5]


def load_checked_model(model_path, ts=0):
    """
    Loads a model from a checkpoint
    """
    if ts != 0:
        path = ts
    else:
        path = model_path
    assert path != ""
    metrics = get_metrics()
    print("Loading existing model: {}.hdf5".format(path))
    file_path = "checkpoints/{}.hdf5".format(path)
    if os.path.exists(file_path):
        print("Loading model from {}".format(file_path))
        model = load_model(file_path, custom_objects={
                               'top2': metrics[1],
                               'top3': metrics[2],
                               'top4': metrics[3],
                               'top5': metrics[4]
                           })
        return model
    else:
        return None


def get_model(x_shape, load_checked, model_path, class_length, training_model):
    """
    Loads a model or creates a new model
    """
    if load_checked and model_path != "":
        model = load_checked_model()
    elif training_model == "NN":
        model = m.neural_network(class_length, x_shape[1])
    elif training_model == "LR":
        model = m.logistic_regression(class_length, x_shape[1])
    else:
        model = m.cnn_model(class_length, (x_shape[1], x_shape[2], x_shape[3]))
    return model


def train_model(model, x_train, y_train, x_validation, y_validation, time_stamp, epochs, batch_size):
    """
    Train the model. This also saves the model as hdf5 depending on validation accuracy.
    """
    # Initiate tensor board
    tensor_board = TensorBoard(log_dir="logs/{}".format(time_stamp))
    checkpoint = ModelCheckpoint("checkpoints/{}.hdf5".format(time_stamp), monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    """
    The following is supposed to pring the model in png. It's not working yet.
    """
    # plot_model(model, to_file='model_plots/{}.png'.format(time_stamp))

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=epochs, batch_size=batch_size,
              callbacks=[tensor_board, checkpoint])
    return model
