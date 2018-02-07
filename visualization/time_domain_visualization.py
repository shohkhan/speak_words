"""
# Python 3.5

Visualize the input data and converted MFCC representations on graphs.

Also, visualize the distribution of the input data lengths.

By Shoaib Khan - Spring 2018
"""

import json
import matplotlib.pyplot as mp
import glob
import numpy as np
import seaborn as sns
from python_speech_features import mfcc

config_json = "../config/config.json"

with open(config_json, "r") as config_file:
    config_data = json.load(config_file)

conversion = config_data["conversion"]

apply_interpolation = conversion["apply_interpolation"]
sampling_frequency = conversion["sampling_frequency"]
highfreq = conversion["highfreq"]
numcep = conversion["numcep"]
nfilt = conversion["nfilt"]
new_length = conversion["new_length"]
minimum_data_size = conversion["minimum_data_size"]


def print_signal(signal, file_path, empty_count):
    if len(signal) == 0:
        print(file_path)
        empty_count += 1
    else:
        mp.plot(signal)
    return empty_count


data_path = "../data/trimmed_v0.2.3/*/*/*.json"

empty_signal1 = 0
empty_signal2 = 0
lengths = []
for file in glob.glob(data_path):
    data = json.load(open(file))
    length = int(data["signal"]["length"])
    label = str(data["label"])
    if length > 0 and label == "a":
        lengths.append(length)
        signal1 = np.array(data["signal"]["1"]).astype(np.float)
        signal2 = np.array(data["signal"]["2"]).astype(np.float)
        empty_signal1 = print_signal(signal1/1000., file, empty_signal1)
        empty_signal2 = print_signal(signal2/1000., file, empty_signal2)
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

        mp.rc('font', **font)
        mp.xlabel("time")
        mp.ylabel("milli volts")
        mp.show()
        m1 = mfcc(signal1, samplerate=sampling_frequency, numcep=numcep, nfilt=nfilt, highfreq=highfreq)
        m2 = mfcc(signal2, samplerate=sampling_frequency, numcep=numcep, nfilt=nfilt, highfreq=highfreq)
        fig, ax = mp.subplots()
        mfcc_data = np.swapaxes(m1, 0, 1)
        cax = ax.imshow(mfcc_data, interpolation='nearest', origin='lower')
        ax.set_title('MFCC')
        mp.xlabel("Time frames")
        mp.ylabel("MFCC bands")
        mp.show()
print("Empty EMG1: " + str(empty_signal1))
print("Empty EMG2: " + str(empty_signal2))

lengths = np.array(lengths)
print (max(lengths), (min(lengths)))

sns.distplot(lengths);

mp.xlim([0, 3000])
mp.hist(lengths, bins=np.arange(lengths.min(), lengths.max()+1))
mp.show()

print(np.mean(lengths), np.median(lengths), str(len(lengths)), str(min(lengths)), str(max(lengths)), str(lengths))
