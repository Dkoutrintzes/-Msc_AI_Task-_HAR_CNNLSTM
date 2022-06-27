from __future__ import print_function
import csv
import os
import time
import numpy as np
from tensorflow.keras.models import Model
#from keras import models
from tensorflow.keras import models
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import warnings
import Generator as load
import tensorflow as tf
import random
import shutil
import csv
config = tf.compat.v1.ConfigProto()
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

twopeople = [11,13,15,17,20,23,25,26]
involveitem = [1,2,3,7,8,9,19,22,27,28,29,34,35,36,37,38,39,45,46,47,48,50]
heavymove = [0,10,14,16,18,21,32,33]
handmovement = [4,5,6,12,24,30,31,40,41,42,43,44,49]

labels = ['bow',
          'brushing_hair',
          'brushing_teeth',
          'check_time',
          'cheer_up',
          'clapping',
          'cross_hands_in_front',
          'drink_water',
          'drop',
          'eat_meal-snack',
          'falling',
          'giving_something_to_other_person',
          'hand_waving',
          'handshaking',
          'hopping',
          'hugging_other_person',
          'jump_up',
          'kicking_other_person',
          'kicking_something',
          'make_a_phone_call-answer_phone',
          'pat_on_back_of_other_person',
          'pickup',
          'playing_with_phone-tablet',
          'point_finger_at_the_other_person',
          'pointing_to_something_with_finger',
          'punching-slapping_other_person',
          'pushing_other_person',
          'put_on_a_hat-cap',
          'put_something_inside_pocket',
          'reading',
          'rub_two_hands_together',
          'salute',
          'sitting_down',
          'standing_up',
          'take_off_a_hat-cap',
          'take_off_glasses',
          'take_off_jacket',
          'take_out_something_from_pocket',
          'taking_a_selfie',
          'tear_up_paper',
          'throw',
          'touch_back',
          'touch_chest',
          'touch_head',
          'touch_neck',
          'typing_on_a_keyboard',
          'use_a_fan',
          'wear_jacket',
          'wear_on_glasses',
          'wipe_face',
          'writing']

# FilePahts for all data

data_labels = os.path.join('ZPLabels')
try:
    os.mkdir('Best_Models')
except:
    print('Exist')

def cr(model,TVfile,actions,dataset,data_labels,dim):
    batch = 1
    if os.path.isfile(os.path.join(data_labels, TVfile)):
        # read from files train and test
        print("Read train files")
        testxNames, testy = load.readFile(os.path.join(data_labels, TVfile))
        print("len train " + str(len(testxNames)))

    Ytest = {testxNames[i]: int(testy[i]) for i in range(len(testy))}

    test_generator = load.DataGenerator(testxNames, Ytest, batch_size=batch, dim=dim, n_channels=3,
                                         n_classes=actions, shuffle=False,dataset=dataset)

    test_loss, test_acc = model.evaluate_generator(
        test_generator,
        steps=len(testy) // batch)
    print('test acc:', test_acc)


    #predictions = model.predict(test_generator)
    #print(predictions)
    # predictions = predictions.argmax(axis=-1)
    # predictions = predictions.tolist()
    return  test_acc

def run_cr(tvfile,exp,actions,dataset,data_labels,modelfolder,dim):
    hmodels = os.listdir(os.path.join(modelfolder,exp))
    bestacc = -199999
    bestname = ''
    for i in hmodels:

        model = models.load_model(os.path.join(modelfolder, exp,i ))
        #model.summary()
        print(i)
        acc = cr(model=model, TVfile=tvfile+'.csv',actions=actions,dataset=dataset,data_labels = data_labels,dim = dim)
        if acc > bestacc:
            print(acc,i)
            bestacc = acc
            bestname = i
        # for m in mdnm:
        #     model = models.load_model(os.path.join(model_folder, m))
        #     #model.summary()
        #     print(m)
        #     cr(model=model, modelname=m, TVfile=tvfile+'.csv',actions=actions)
    shutil.copy2(os.path.join(modelfolder, exp,bestname ),'Best_Models')
    os.rename(os.path.join('Best_Models',bestname ),os.path.join('Best_Models',exp+tvfile ))
if __name__ == '__main__':

    # run_cr(tvfile='L', exp='MRD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',modelfolder='Models',dim=(149, 25) )
    # run_cr(tvfile='CST', exp='CSD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',modelfolder='Models',dim=(149, 25) )

    run_cr(tvfile='L', exp='MRLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',modelfolder='Models',dim=(199, 25) )
    run_cr(tvfile='CST', exp='CSLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',modelfolder='Models',dim=(199, 25) )

    run_cr(tvfile='L', exp='MRLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',modelfolder='Models',dim=(199, 50) )
    run_cr(tvfile='CST', exp='CSLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',modelfolder='Models',dim=(199, 50) )





