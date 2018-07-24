# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle

sys.path.append("/home/wkwang/workstation/experiment/src")


def curve(data, name, place):
    X = np.arange(0, len(data), 200)
    Y = list()
    std_Y = list()
    for start in X:
        Y.append(np.mean(data[start: start + 200 if (start + 200) < len(data) else len(data)]))
        std_Y.append(np.std(data[start: start + 200 if (start + 200) < len(data) else len(data)]))
    plt.subplot(place)
    plt.errorbar(X, Y, std_Y)
    plt.title(name)


def curve_no_mean(data, name):
    X = np.arange(0, len(data))
    Y = np.array(data)
    data = (X, Y)
    # plt.subplot(place)
    plt.scatter(X, Y, s=1)
    plt.title(name)


if __name__ == "__main__":
    data_file = os.path.join("/home/wkwang/workstation/experiment/src", "Continuous_VAE", "debug", "loss.log")
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    for i, (key, value) in enumerate(data.items()):
        curve(value, key, int(str("32") + str(i + 1)))
    # curve_no_mean(data["acc"], "acc")
    plt.show()
