import xlrd
import os
import numpy as np

def multiD_MinMax(data, min_value=None, max_value=None):
    def MinMaxScaler(data, min, max):
        return (data - min) / (max - min + 1e-7)


    if min_value is None and max_value is None:
        min_list = []
        max_list = []

        for one_data in data:
            min_list.append(np.min(one_data, 0))
            max_list.append(np.max(one_data, 0))
        min_value = np.min(min_list, 0)
        max_value = np.max(max_list, 0)

    for idx, one_data in enumerate(data):
        data[idx] = MinMaxScaler(np.array(one_data), np.array(min_value), np.array(max_value))

    return data, min_value, max_value

def MinMaxScaler(data, min_v=None, max_v=None):
    if min_v is None and max_v is None:
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)

    else:
        numerator = data - min_v
        denominator = max_v - min_v

    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def ReverseMInMax(data, min_v, max_v):
    return (max_v-min_v) * np.array(data) + min_v