"""
# Python 3.5

This file reads the config.json file and runs the classification algorithms over the training dataset and
evaluates on the testing dataset.

By Shoaib Khan - Spring 2018
"""


import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import json
from time import time
import utils as u
import csv

K.set_image_dim_ordering('tf')

"""
Set the numpy.random seed to make the experiments reproducible 
"""
np.random.seed(7)

"""
Set the time-stamp to track the results
"""
time_stamp = time()

"""
Load the fields from the config file
"""
config_json = "../config/config.json"
config_data = json.load(open(config_json, "r"))

training_model = config_data["model"]

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
    print("Using Model: {}".format(training_model))
    u.create_required_directories()

    """
    Step 1: Get the data
    """
    x_train, train_paths = u.read_files(train_data_path, training_subjects, folder_name)
    x_test, test_paths = u.read_files(test_data_path, testing_subjects, folder_name)
    x_validation, val_paths = u.read_files(val_data_path, testing_subjects, folder_name)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_validation = np.array(x_validation)
    print(x_train.shape, x_test.shape, x_validation.shape)

    """
    Step 1.1: Shuffle the training data 
    """
    np.random.shuffle(x_train)

    """
    Step 1.2: Split x and y values
    """
    x_train, y_train, _ = u.split_x_y(x_train)
    x_test, y_test, y_test_labeled = u.split_x_y(x_test)
    x_validation, y_validation, _ = u.split_x_y(x_validation)

    """
    Step 2: Normalize the data
    """
    mean_array = np.mean(x_train, axis=0)
    x_train -= mean_array
    x_test -= mean_array
    x_validation -= mean_array

    max_value = np.max(x_train)
    x_train /= float(max_value)
    x_test /= float(max_value)
    x_validation /= float(max_value)

    normalizing_values = {
        "mean_array": mean_array,
        "max_value": max_value
    }

    """
    Step 2.2: Save the normalizing values for analysis
    """
    u.save_object(normalizing_values, "normalizers/{}".format(time_stamp))

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    if training_model == "NN" or training_model == "LR":
        """
        For Fully Connected Neural Nets and Logistic Regression, the dimension is required to be changed 
        """
        dimension = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
        x_train = x_train.reshape(x_train.shape[0], dimension).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], dimension).astype('float32')
        x_validation = x_validation.reshape(x_validation.shape[0], dimension).astype('float32')

    """
    Step 3: Get the model.
    """
    model = u.get_model(x_train.shape, load_checked, model_path, len(labels), training_model)

    """
    Step 3.1: Compile model
    """
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=u.get_metrics())
    print(model.summary())

    """
    Step 4: Train the model.
    """
    if not predict_only:
        model = u.train_model(model, x_train, y_train, x_validation, y_validation, time_stamp, epochs, batch_size)

    """
    Step 4.1: After training reload the model depending on the best performance on the validation dataset
    """
    if reload_model:
        model = u.load_checked_model(model_path, time_stamp)

    """
    Step 5: Evaluate the model
    """
    # Get the predictions on test set
    prediction_y = model.predict(x_test)

    pred = np.array(prediction_y)
    pred = pred.argmax(1)
    pred = np.array(pred).astype(int)

    y_test_labeled = np.array(y_test_labeled).astype(int)

    # Get and save the confusion matrix
    cf = confusion_matrix(y_test_labeled, pred)
    with open("confusion_matrix/{}".format(time_stamp), "w") as file:
        sentences = []
        for line in cf:
            s = ""
            for item in line:
                s += "{},".format(item)
            sentences.append(s + "\n")
        file.writelines(sentences)
    print(labels)

    # Show the precision, recall and F1 measure for all the classes
    print(classification_report(y_test_labeled, pred))

    # Get and save the top-5 predictions for each word in test
    p_array = []
    for i in range(len(prediction_y)):
        p = prediction_y[i]
        top5 = np.argpartition(p, -5)[-5:]
        out = [labels[y_test_labeled[i]]]
        for j in top5:
            probability = p[j]
            prediction_label = labels[j]
            out.append((prediction_label, probability))
        p_array.append(out)
    with open("model_plots/output_{}.csv".format(time_stamp), 'w') as resultFile:
        wr = csv.writer(resultFile)
        wr.writerows(p_array)

    # Calculate the accuracy on the test dataset
    scores = model.evaluate(x_test, y_test)
    print(scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    if load_checked and model_path != "":
        print("Initiating Model: " + model_path)

    # Print some of the parameters related to the process
    print(train_paths)
    print(test_paths)
    print(folder_name, epochs, batch_size, learning_rate, time_stamp)


run_classifier()
