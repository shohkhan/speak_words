"""
# Python 3.5

This script reads config.json file. Then depending on the configuration settings, converts the input raw signals to
their MFCC representations.

By Shoaib Khan - Spring 2018
"""

import glob
import numpy as np
from python_speech_features import mfcc
import os
import h5py
from helper import interpolate, pad
import matplotlib.pyplot as mp
import json
import librosa
import noise_generator as ng
from matplotlib.legend_handler import HandlerLine2D
from sklearn.preprocessing import normalize
"""
First, read the config file. The details about the variables read from the config file are provided in the readme of 
the directory containing the config.json file.
"""
config_json = "../config/config.json"
with open(config_json, "r") as config_file:
    config_data = json.load(config_file)

data_path = config_data["data_path"]

conversion = config_data["conversion"]

test_fraction = conversion["test_fraction"]
conversion_type = conversion["conversion_type"]
apply_interpolation = conversion["apply_interpolation"]
apply_padding = conversion["apply_padding"]
apply_stack = conversion["apply_stack"]
show_plot = conversion["show_plot"]
skip_second = conversion["skip_second"]
sampling_frequency = conversion["sampling_frequency"]
highfreq = conversion["highfreq"]
numcep = conversion["numcep"]
nfilt = conversion["nfilt"]
new_length = conversion["new_length"]
minimum_data_size = conversion["minimum_data_size"]

noises = conversion["noises"]
noise_coefficients = conversion["noise_coefficients"]

apply_shift = conversion["apply_shift"]
apply_stretch = conversion["apply_stretch"]

labels = config_data["labels"]
subjects = config_data["available_subjects"]

des_string = "{}_{}_{}_{}_{}_{}_{}_{}".format(conversion_type, sampling_frequency, highfreq, numcep, nfilt,
                                              new_length, minimum_data_size, len(labels))
if apply_interpolation:
    des_string += "_Interpolation"
elif apply_padding:
    des_string += "_Padding"
else:
    des_string += "_VariableLength"
lengths = []
lengths_time = []


def add_noise(signal_array, coefficient, name=None, print_noise_sum=False):
    """
    This function adds noise to any input signal.
    Help from https://www.kaggle.com/CVxTz/audio-data-augmentation
    :param signal_array: input signal.
    :param coefficient:
    :param name: name of the noise.
    :param print_noise_sum: For debigging. Prints the sum of the noise.
    :return:
    """
    if name == "white":
        noise = ng.white(len(signal_array))
    elif name == "pink":
        noise = ng.pink(len(signal_array))
    elif name == "blue":
        noise = ng.blue(len(signal_array))
    elif name == "brown":
        noise = ng.brown(len(signal_array))
    elif name == "violet":
        noise = ng.violet(len(signal_array))
    else:
        noise = np.random.randn(len(signal_array))
    if print_noise_sum:
        print(sum(noise))
    modified_signal_array = signal_array + coefficient * noise
    return modified_signal_array


def stretch(signal_array, rate=1.0):
    """
    This function stretches the signals in the time domain.
    External library used: librosa
    :param signal_array:
    :param rate:
    :return:
    """
    modified_signal_array = librosa.effects.time_stretch(signal_array, rate)
    return modified_signal_array


def get_mfcc(signal1, signal2, count, label, uuid, subject):
    """
    This function takes the input signals and converts them to MFCCs and saves the converted representations to
    a destination directory in the hdf5 format.
    :param signal1:
    :param signal2:
    :param count:
    :param label:
    :param uuid:
    :param subject:
    :return:
    """
    destination = "../data/{}/".format(des_string)
    if show_plot:
        line1, = mp.plot(signal1/1000., label="Channel 1")
        _, = mp.plot(signal2/1000., label="Channel 2")
        mp.rcParams.update({'font.size': 20})
        mp.locator_params(axis='y', nticks=3)
        mp.yticks()
        mp.xlabel("Samples (4000 per second)")
        mp.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
        mp.ylabel("Milli volts")
        mp.show()
    m1_array = [signal1]
    m2_array = [signal2]

    """
    One in every n is in test and one in every n is in validation. Other (n-2) are in training.
    """
    test_data = (count % test_fraction) == 1
    validation_data = (count % test_fraction) == 2

    if not test_data and not validation_data:
        """
        Data augmentation step. Data augmentation is only done for the training dataset, not for testing or validation.
        """
        for noise in noises:
            for coefficient in noise_coefficients:
                # Add moise to the signal
                m1_array.append(add_noise(signal1, float(coefficient), noise))
                m2_array.append(add_noise(signal2, float(coefficient), noise))

        if apply_shift:
            # SHIFT RIGHT
            m1_array.append(np.roll(signal1, int(len(signal1) * 0.05)))
            m2_array.append(np.roll(signal2, int(len(signal2) * 0.05)))

            # SHIFT LEFT
            m1_array.append(np.roll(signal1, int(len(signal1) * 0.05 * -1)))
            m2_array.append(np.roll(signal2, int(len(signal2) * 0.05 * -1)))

        if apply_stretch:
            # Stretch the signal
            m1_array.append(stretch(signal1, 0.8))
            m2_array.append(stretch(signal2, 0.8))

            m1_array.append(stretch(signal1, 1.2))
            m2_array.append(stretch(signal2, 1.2))

    for i in range(len(m1_array)):
        """
        Convert the original and augmented data to their MFCC representations.
        """
        signal1 = m1_array[i]
        signal2 = m2_array[i]

        if len(signal1) < 1:
            continue

        m1 = mfcc(signal1, samplerate=sampling_frequency, numcep=numcep, nfilt=nfilt, highfreq=highfreq)
        m2 = mfcc(signal2, samplerate=sampling_frequency, numcep=numcep, nfilt=nfilt, highfreq=highfreq)

        if i == 0:
            lengths.append(len(m1))

        if show_plot:
            print(len(m1), len(m1[0]))

        if apply_interpolation:
            # Use interpolation to make the converted data have the same sizes
            m1 = interpolate(m1, new_length)
            m2 = interpolate(m2, new_length)

        if apply_padding:
            # Use padding to make the converted data have the same sizes
            m1 = pad(m1, new_length, True)
            m2 = pad(m2, new_length, True)

        if apply_stack:
            # Stack the signals to make them comparable to multi-channel inputs to the models. Helpful for CNNs.
            signal = np.dstack((m1, m2))
        else:
            m1 = m1.T
            m2 = m2.T
            signal = np.concatenate((m1, m2))
            signal = signal.T
            normalize(signal, axis=0)

        if show_plot:
            """
            Show converted MFCC representations
            """
            if i == 1:
                fig, ax = mp.subplots()
                mfcc_data = np.swapaxes(m1, 0, 1)
                ax.imshow(mfcc_data, interpolation='nearest', origin='lower')
                ax.set_title('MFCC Representation of Channel 1')
                mp.xlabel("Time Frames")
                mp.ylabel("Mel Bands")
                mp.show()

                fig, ax = mp.subplots()
                mfcc_data = np.swapaxes(m2, 0, 1)
                ax.imshow(mfcc_data, interpolation='nearest', origin='lower')
                ax.set_title('MFCC Representation of Channel 2')
                mp.xlabel("Time Frames")
                mp.ylabel("Mel Bands")
                mp.show()

        """
        Round the converted signals to 4 decimal places only. Made unreachable for now.
        """
        if False:
            signal = np.round(signal, decimals=4)

        """
        Save the files
        """
        data = {
            "label": labels.index(label),
            "signal": signal
            }

        if not os.path.exists(destination):
            os.makedirs(destination)
        sub_destination = destination + subject + "/"
        if not os.path.exists(sub_destination):
            os.makedirs(sub_destination)
        if not os.path.exists(sub_destination + label):
            os.makedirs(sub_destination + label)
        if not os.path.exists(sub_destination + label + "/test"):
            os.makedirs(sub_destination + label + "/test")
        if not os.path.exists(sub_destination + label + "/validation"):
            os.makedirs(sub_destination + label + "/validation")

        if test_data:
            file_location = "{}{}/test/{}_{}_{}.hdf5".format(sub_destination, label, uuid, str(count), str(i))
        elif validation_data:
            file_location = "{}{}/validation/{}_{}_{}.hdf5".format(sub_destination, label, uuid, count, str(i))
        else:
            file_location = "{}{}/{}_{}_{}.hdf5".format(sub_destination, label, uuid, str(count), str(i))

        with h5py.File(file_location, 'w') as f:
            for k, v in data.items():
                f.create_dataset(k, data=np.array(v, dtype=float))
    return


def convert_to_mfcc():
    """
    For each of the subjects considered, this function takes the raw signals from the corresponding directory and
    converts them to their MFCC representations.
    """
    count = 0
    for s in subjects:
        print("Subject: {}".format(s))
        for file in glob.glob(data_path.format(s)):
            data = json.load(open(file))
            length = int(data["signal"]["length"])
            if length >= minimum_data_size:
                lengths_time.append(length)
                label = data["label"]
                if label in labels:
                    count += 1
                    uuid = data["uuid"]
                    signal1 = np.array(data["signal"]["1"]).astype(np.float)
                    signal2 = np.array(data["signal"]["2"]).astype(np.float)
                    get_mfcc(signal1, signal2, count, label, uuid, s)

    print("Total processed: " + str(count))
    print("Destination Folders: " + str(des_string))

    if show_plot:
        """
        Show the histograms of the distributions of lengths of the signals in the time domain and MFCC domain 
        """
        print(len(lengths), np.mean(lengths), np.median(lengths), max(lengths), (min(lengths)))
        mp.hist(lengths)
        mp.show()

        print(len(lengths_time), np.mean(lengths_time), np.median(lengths_time), max(lengths_time), (min(lengths_time)))
        mp.hist(lengths_time)
        mp.show()

    return des_string


convert_to_mfcc()
