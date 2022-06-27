from __future__ import print_function
import os
from posixpath import split
import time
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
import random
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import Generator as load
import pathlib
import Model as md
import tensorflow as tf
from natsort import natsorted
import os
import time

import tensorflow.keras as  keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from natsort import natsorted
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
from CNNLSTM_model import create_LSTM_CNN,Default_Model,create_LSTM_CNN_2P
config = tf.compat.v1.ConfigProto()
# config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def saperate_train_valid(xNames, y,actions):
    trainxNames = []
    trainy = []
    tempdata = []
    validxNames = []
    validy = []
    c = []
    d = len(xNames)*0.1
    for label in range(0,actions):
        #print(y,str(label))
        tempdata = [i for i in range(len(y)) if y[i] == str(label)]
        if len(tempdata) == 0:
            continue
        #print(len(tempdata))

        for i in range(int(d//60)):
            c.append(np.random.choice(tempdata))
    c = natsorted(c)
    for i in range(len(y)):
        if i in c:
            validxNames.append(xNames[i])
            validy.append((y[i]))
        else:
            trainxNames.append(xNames[i])
            trainy.append((y[i]))
    return trainxNames, trainy, validxNames, validy

def sep(xNames,y,actions):
    trainxNames = []
    trainy = []

    validxNames = []
    validy = []

    tempdata = [i for i in range(len(y)) if xNames[i].split('_')[2] == 'R']

    for i in range(len(y)):
        if i in tempdata:
            validxNames.append(xNames[i])
            validy.append((y[i]))
        else:
            trainxNames.append(xNames[i])
            trainy.append((y[i]))
    return trainxNames, trainy, validxNames, validy


def split_train_valid(xNames, y,actions,validsplit):
    trainxNames = []
    trainy = []
    validxNames = []
    validy = []

    for i in range(len(xNames)):
        if xNames[i].split('_')[2] == validsplit:
            validxNames.append(xNames[i])
            validy.append((y[i]))
        else:
            trainxNames.append(xNames[i])
            trainy.append((y[i]))

    return trainxNames, trainy, validxNames, validy

        


twopeople = [11,13,15,17,20,23,25,26]
involveitem = [1,2,3,7,8,9,19,22,27,28,29,34,35,36,37,38,39,45,46,47,48,50]
heavymove = [0,10,14,16,18,21,32,33]
handmovement = [4,5,6,12,24,30,31,40,41,42,43,44,49]
all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]


epoch = 150
batch = 32





def main(model, TVfile='', exp='Un',actions=60,dataset ='',data_labels='',save_file='Models',dim=(199, 25) ):

    try:
        os.mkdir(save_file)
    except:
        print('File exist')

    optimizer = keras.optimizers.Adam()
    #optimizer = keras.optimizers.RMSprop()
    #optimizer = keras.optimizers.SGD()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    start = time.perf_counter()
    weight_decay = 0.005

    # Load From csv file the paths and labels for the Train Data
    # Load From csv file the paths and labels for the Train Data
    if os.path.isfile(os.path.join(data_labels, TVfile + '.csv')):
        # read from files train and test
        # print("Read train files")
        xNames, y = load.readFile(os.path.join(data_labels, TVfile + '.csv'))
        #print("len train " + str(len(xNames)))

    #for each in sidestorun:
    trainxNames, trainy, validxNames, validy = saperate_train_valid(xNames, y, actions)
    print("len train " + str(len(trainxNames)))
    print("len valid " + str(len(validxNames)))
    Ytrain = {trainxNames[i]: int(trainy[i]) for i in range(len(trainy))}
    Yvalid = {validxNames[i]: int(validy[i]) for i in range(len(validy))}


    train_generator = load.DataGenerator(trainxNames, Ytrain, batch_size=batch, dim=dim, n_channels=3,
                                        n_classes=actions, shuffle=True, dataset = dataset)

    valid_generator = load.DataGenerator(validxNames, Yvalid, batch_size=batch, dim=dim, n_channels=3,
                                        n_classes=actions, shuffle=True, dataset = dataset)

    steps_per_epoch = len(trainxNames) // batch
    val_steps = len(validxNames) // batch

    m = md.models(batch, weight_decay, steps_per_epoch, epoch)
    m.train(model, train_generator, valid_generator, epoch, steps_per_epoch, val_steps, save_file, exp)




if __name__ == '__main__':

    # main(model = Default_Model((149,25,3),60), TVfile='MR', exp='MRD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',save_file='Models',dim=(149, 25) )
    # main(model = Default_Model((149,25,3),60), TVfile='CS', exp='CSD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',save_file='Models',dim=(149, 25) )

    # main(model = create_LSTM_CNN((199,25,3),60), TVfile='MR', exp='MRLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',save_file='Models',dim=(199, 25) )
    # main(model = create_LSTM_CNN((199,25,3),60), TVfile='CS', exp='CSLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',save_file='Models',dim=(199, 25) )

    # main(model = create_LSTM_CNN((199,50,3),60), TVfile='MR', exp='MRLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',save_file='Models',dim=(199, 50) )
    main(model = create_LSTM_CNN((199,50,3),60), TVfile='CS', exp='CSLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',save_file='Models',dim=(199, 50) )



