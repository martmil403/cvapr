from keras.models import Sequential
import data as data
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K


def calculateq3(ypredicted, ytest):
    for i in range(ypredicted.shape[0]):
        # G
        if ypredicted[i] == 7:
            ypredicted[i] = 0
        elif ypredicted[i] == 2:
            ypredicted[i] = 1
        else:
            ypredicted[i] = 2
    for i in range(ypredicted.shape[0]):
        # G
        if ytest[i] == 7:
            ytest[i] = 0
        elif ytest[i] == 2:
            ytest[i] = 1
        else:
            ytest[i] = 2
    right = 0
    wrong = 0
    for i in range(ypredicted.shape[0]):
        if ypredicted[i] == ytest[i]:
            right += 1
        else:
            wrong += 1
    return right/float(right+wrong)
