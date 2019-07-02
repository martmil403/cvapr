from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Conv2D, Dropout, LSTM, BatchNormalization
import data as data
import numpy as np
import tensorflow as tf
import keras
import calculateq3 as q3
from keras import backend as K
from keras.models import load_model


def calcq3(model, val_x, val_y):
    ypredicted = model.predict(val_x)
    ypredicted = np.argmax(ypredicted, axis=-1)
    accq3 = q3.calculateq3(ypredicted, val_y)
    print(accq3)
    return accq3


addchannel = False
batch_size = 64

tensorboard = keras.callbacks.TensorBoard()
checkpoint = keras.callbacks.ModelCheckpoint(
    './models/cnn-{val_acc:.4f}.hdf5', monitor='val_acc', save_best_only=True, verbose=True)

for i in range(4):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=(22*5), strides=22,
                     activation='relu',
                     input_shape=(22*15, 1)))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=(3), strides=1,
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(64, kernel_size=(3), strides=1,
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(LSTM(256))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    trainx, trainy, val_x, val_y = data.crossvalidation(i, 4)
    trainx, trainy = data.dataconversion(trainx.shape[0], trainx, trainy)
    trainx = np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))

    val_x, val_y = data.dataconversion(val_x.shape[0], val_x, val_y)
    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))
    model.fit(trainx, trainy, epochs=1, batch_size=32, validation_data=(
        val_x, val_y), callbacks=[tensorboard, checkpoint])
    calcq3(model, val_x, val_y)

