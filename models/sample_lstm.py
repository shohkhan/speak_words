
"""
# Python 3.5

This is a proof of concept that LSTMs also work for speech recognition.
However, as this is a word recognition project, the CNNs are performing better for now.
There are a lot of room left to improve on the RNNs though.

By Shoaib Khan - Spring 2018
"""
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
import models as m
import utils as u
import json
from time import time
from sklearn.metrics import classification_report, confusion_matrix
K.set_image_dim_ordering('tf')

np.random.seed(7)
time_stamp = time()
config_json = "../config/config.json"
config_data = json.load(open(config_json, "r"))

training_subjects = config_data["training_subjects"]
testing_subjects = config_data["testing_subjects"]
labels = config_data["labels"]

config = config_data["training"]

model_path = config["model_checkpoint"]
load_checked = config["load_checked"]
predict_only = config["predict_only"]
reload_model = config["reload_model"]

folder_name = config["data_folder_name"]

train_data_path = config["train_data_path"]
test_data_path = config["test_data_path"]
val_data_path = config["val_data_path"]

learning_rate = config["learning_rate"]
epochs = config["epochs"]
batch_size = config["batch_size"]


def run_classifier():
    u.create_required_directories()

    """
    Get the data
    """
    x_train, train_paths = u.read_files(train_data_path, training_subjects, folder_name)
    x_test, test_paths = u.read_files(test_data_path, testing_subjects, folder_name)
    x_validation, val_paths = u.read_files(val_data_path, testing_subjects, folder_name)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_validation = np.array(x_validation)
    print(x_train.shape, x_test.shape, x_validation.shape)

    np.random.shuffle(x_train)

    x_train, y_train, _ = u.split_x_y(x_train)
    x_test, y_test, y_test_labeled = u.split_x_y(x_test)
    x_validation, y_validation, _ = u.split_x_y(x_validation)

    # x_train = u.pad_rows(x_train, 60, True)
    # x_test = u.pad_rows(x_test, 60, True)
    # x_validation = u.pad_rows(x_validation, 60, True)

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    """
    Get the model
    """

    # model = m.simple_rnn_model(len(labels), x_train.shape)
    model = m.simple_rnn_model(len(labels), len(x_train[0]))

    print(model.summary())

    adam = Adam(lr=learning_rate)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Fit the model
    model = u.train_model(model, x_train, y_train, x_validation, y_validation, time_stamp, epochs, batch_size)

    """
    Evaluate the model
    """
    prediction_y = model.predict(x_test)
    pred = np.array(prediction_y)
    pred = pred.argmax(1)
    pred = np.array(pred).astype(int)

    y_test_labeled = np.array(y_test_labeled).astype(int)

    cf = confusion_matrix(y_test_labeled, pred)
    print(cf)
    with open("confusion_matrix/{}".format(time_stamp), "w") as file:
        sentences = []
        for line in cf:
            s = ""
            for item in line:
                s += "{},".format(item)
            sentences.append(s + "\n")
        file.writelines(sentences)
    print(labels)
    print(classification_report(y_test_labeled, pred))

    scores = model.evaluate(x_test, y_test)
    print(scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    if load_checked and model_path != "":
        print("Initiating Model: " + model_path)
    print(train_paths)
    print(test_paths)
    print(folder_name, epochs, batch_size, learning_rate, time_stamp)


run_classifier()


