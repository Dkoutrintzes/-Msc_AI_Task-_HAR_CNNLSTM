from __future__ import print_function
import os
import time
import numpy as np
import random
import tensorflow as tf

import tensorflow.keras as  keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
import warnings
from tensorflow.keras.layers import TimeDistributed, LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda

# def create_LSTM_CNN(input_shape,action_size):
#     model = models.Sequential()
#     model.add(
#         layers.Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                         kernel_constraint=max_norm(max_value=2), input_shape=input_shape))
#     # model.add(LeakyReLU(alpha=0.3))
#     model.add(layers.MaxPooling2D((2, 2)))

#     model.add(
#         layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                         kernel_constraint=max_norm(max_value=2)))
#     # model.add(LeakyReLU(alpha=0.3))

#     model.add(
#         layers.Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                         kernel_constraint=max_norm(max_value=2)))
#     # model.add(LeakyReLU(alpha=0.3))

#     model.add(layers.MaxPooling2D((2, 2)))

#     model.add(
#         layers.Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
#                         kernel_constraint=max_norm(max_value=2)))
#     # model.add(LeakyReLU(alpha=0.3))

#     model.add(layers.MaxPooling2D((2, 2)))


#     model.add(TimeDistributed(Flatten(data_format = 'channels_first')))
#     model.add(LSTM(128, return_sequences=True))
#     # model.add(TimeDistributed(layers.Dropout(0.3)))
#     # model.add(LSTM(256, return_sequences=True))



#     model.add(layers.Flatten())
#     model.add(layers.Dense(512, activation='relu'))
#     # model.add(LeakyReLU(alpha=0.3))
#     model.add(layers.Dropout(0.3))
#     model.add(layers.Dense(128, activation='relu', name='feature'))
#     model.add(layers.Dense(action_size, activation='softmax'
#                             ))
        
#     return model
def create_LSTM_CNN(input_shape,action_size):
    model = models.Sequential()
    model.add(
        layers.Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2), input_shape=input_shape))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(
        layers.Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(256, (3, 3), padding='valid', activation='tanh', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(layers.MaxPooling2D((2, 2)))


    model.add(TimeDistributed(Flatten(data_format = 'channels_first')))
    model.add(LSTM(256, return_sequences=True,
                       ))
    # model.add(TimeDistributed(layers.Dropout(0.3)))
    model.add(LSTM(256, return_sequences=True,
                       ))



    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu', name='feature'))
    model.add(layers.Dense(action_size, activation='sigmoid'
                            ))
        
    return model


def create_LSTM_CNN_2P(input_shape,action_size):
    model = models.Sequential()
    model.add(
        layers.Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2), input_shape=input_shape))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(
        layers.Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(256, (3, 3), padding='valid', activation='tanh', kernel_regularizer=regularizers.l2(0.001),
                        kernel_constraint=max_norm(max_value=2)))
    # model.add(LeakyReLU(alpha=0.3))

    model.add(layers.MaxPooling2D((2, 2)))


    model.add(TimeDistributed(Flatten(data_format = 'channels_first')))
    model.add(LSTM(256, return_sequences=True,
                       ))
    # model.add(TimeDistributed(layers.Dropout(0.3)))
    model.add(LSTM(256, return_sequences=True,
                       ))



    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    # model.add(LeakyReLU(alpha=0.3))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu', name='feature'))
    model.add(layers.Dense(action_size, activation='sigmoid'
                            ))
        
    return model

def create_MOS_CNN(input_shape,action_size):
    model = models.Sequential()
    model.add(
        layers.Conv2D(16, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                      kernel_constraint=max_norm(max_value=2), input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                      kernel_constraint=max_norm(max_value=2)))

    
    model.add(
        layers.Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                      kernel_constraint=max_norm(max_value=2)))

    model.add(layers.MaxPooling2D((2, 2)))


    model.add(
        layers.Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001),
                      kernel_constraint=max_norm(max_value=2)))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', name='feature'))
    model.add(layers.Dense(action_size, activation='sigmoid'
                           ))
    return model


# model = create_LSTM_CNN_2P((199, 50, 3),60)
# model = create_LSTM_CNN((199, 25, 3),60)
# model.summary()
