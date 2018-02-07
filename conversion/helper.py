"""
# Python 3.5

This contains helper functions required to make the convert_to_mfcc.py script work.

By Shoaib Khan - Spring 2018
"""

import numpy as np


def interpolate(signal, new_length):
    """
    This function uses linear interpolation to convert the inputs to an array of size new_length.
    Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html
    """
    assert len(signal) > 1 and len(signal[0]) > 1
    current_length = len(signal)
    signal = np.array(signal).T
    new_signal = []
    x_array = get_x_array(current_length, new_length)

    for l in range(len(signal)):
        fp = signal[l]
        xp = list(range(current_length))
        new_f = np.interp(x_array, xp, fp)
        new_signal.append(new_f)

    signal = np.array(new_signal).T
    return signal


def get_x_array(current_length, new_length):
    """
    This is used by the Interpolate function
    """
    d = current_length / (new_length - 1)
    x_array = []
    for i in range(new_length):
        if i == new_length - 1:
            x_array.append(current_length)
        else:
            x_array.append(i * d)
    return x_array


def pad(signal, new_length, end):
    """
    This makes the input size to have a specific size.

    end: If true the padding will be done at the end of the signal. If false the padding will be done at the end and
    at the beginning of the signal - to make the padding symmetric.
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
