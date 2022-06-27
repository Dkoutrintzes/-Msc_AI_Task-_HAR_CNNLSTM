import numpy as np
import tensorflow as tf
import tensorflow.keras as ker
# from tensorflow._api.v1 import keras


import math
from tensorflow.keras.models import Sequential
# tensorflow._api.v1.keras
from tensorflow.keras import Input, Model

# import layers modules
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import (Conv2D, MaxPooling3D, Conv3D, ConvLSTM2D,
                                     MaxPooling2D, AveragePooling3D, ZeroPadding3D, BatchNormalization, Activation)

from tensorflow.keras.regularizers import l2

# from tensorflow.keras.utils import plot_model

# import optimizers
from tensorflow.keras.optimizers import SGD, Adam
# from keras.optimizers.schedules import PolynomialDecay
# import essential tools for trainning
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, LearningRateScheduler
from time import time
import os
import matplotlib.pyplot as plt
import itertools


class models:

    def __init__(self, batch, weight_decay, steps_per_epoch, nb_epoch):
        self.batch = batch
        self.weight_decay = weight_decay
        self.steps_per_epoch = steps_per_epoch
        self.nb_epoch = nb_epoch

    def step_decay(self, epoch):
        initial_lrate = 0.001
        drop = 0.1
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def lr_polynomial_decay(self, global_step):
        learning_rate = 0.00001
        end_learning_rate = 0.000002
        decay_steps = self.steps_per_epoch * self.nb_epoch
        power = 0.9
        p = float(global_step) / float(decay_steps)
        lr = (learning_rate - end_learning_rate) * np.power(1 - p, power) + end_learning_rate
        return lr

        # input_shape = (20, 120, 120, 3)

    def train(self, model, train_generator, val_generator, nb_epoch, steps_per_epoch, val_steps, save_model_path, exp):

        # define a polynomial decay scheduler

        lr = tf.keras.optimizers.schedules.PolynomialDecay(
                    0.00003,
                    200,
                    end_learning_rate=0.000001,
                    power=0.9,
                    cycle=False,
                    name=None,
                )
        lrate = LearningRateScheduler(lr, steps_per_epoch)
        #lrate = LearningRateScheduler(self.lr_polynomial_decay, steps_per_epoch)
        # optimizer
        # optimizer = SGD( momentum=0.9, decay=0)
        # %tensorboard --logdir  media/skylos/Storage/Cod/adms_extra/NewImages/Models/logs/
        # optimizer = Adam(lr=1e-5, decay=1e-6)
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
        # metrics=["accuracy"])

        tb = TensorBoard(log_dir=os.path.join(save_model_path, 'logs', '{}'.format(time())), histogram_freq=0,
                         write_graph=True)

        early_stopper = EarlyStopping(patience=50)

        if not os.path.isdir(os.path.join(save_model_path, exp)):
            os.mkdir(os.path.join(save_model_path, exp))

        checkpointer = ModelCheckpoint(
            filepath=os.path.join(save_model_path, exp, '{epoch:02d}-{val_loss:.1f}.h5'),
            verbose=1,
            save_best_only=False)
        # train the network
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=nb_epoch,
                                      verbose=1,
                                      callbacks=[tb, checkpointer, lrate],
                                      validation_data=val_generator,
                                      validation_steps=val_steps,
                                      )

        # Create loss accuracy diagram
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print(acc, val_acc, val_loss)
        # make folder to save the diagrams
        if not os.path.isdir(os.path.join(save_model_path, "conf_matrix")):
            os.mkdir(os.path.join(save_model_path, "conf_matrix"))

        nameloss = "model train vs validation loss"
        nameacc = "model train vs validation accuracy"
        # for loss
        if not os.path.isdir(os.path.join(save_model_path, "diagrams")):
            os.mkdir(os.path.join(save_model_path, "diagrams"))

        fig_folder = os.path.join(save_model_path, "diagrams", exp + "_" + "loss")
        self.createplot(loss, val_loss, nameloss, fig_folder, "loss")
        # for accuracy
        fig_folder = os.path.join(save_model_path, "diagrams", exp + "_" + "accuracy")

        self.createplot(acc, val_acc, nameacc, fig_folder, "accuracy")

        modelname = exp + '.h5'
        model.save(os.path.join(save_model_path, exp, modelname))

    def createplot(self, datatrain, dataval, figurename, name, typeofdata):
        # print(datatrain)
        plt.figure(figurename)
        plt.plot(datatrain)
        plt.plot(dataval)
        plt.title(figurename)
        plt.ylabel(typeofdata)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(name + ".png")
        plt.cla()
        plt.clf()
        plt.close()
