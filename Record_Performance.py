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
import seaborn as sn
import tensorflow as tf
import random
import cv2
from PIL import ImageFont
import csv
config = tf.compat.v1.ConfigProto()
#config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


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


def cr(modelname,tvfile,actions,dataset,data_labels,modelfolder,dim):
    model = models.load_model(os.path.join(modelfolder,modelname+tvfile))
    model.summary()
    print(model.layers[-2].name)
    batch = 1
    if os.path.isfile(os.path.join(data_labels, tvfile+'.csv')):
        # read from files train and test
        print("Read train files")
        testxNames, testy = load.readFile(os.path.join(data_labels, tvfile+'.csv'))
        print("len train " + str(len(testxNames)))

    Ytest = {testxNames[i]: int(testy[i]) for i in range(len(testy))}

    test_generator = load.DataGenerator(testxNames, Ytest, batch_size=batch, dim=dim, n_channels=3,
                                         n_classes=actions, shuffle=False,dataset=dataset)

    preds = model.predict(test_generator)
    preds = preds.argmax(axis=-1)
    preds = preds.tolist()
    # pprint(preds)
    # print(preds)
    # print(np.shape(testtrues),np.shape(preds))]
    trues = []
    for i in range(len(testy)):
        trues.append(int(testy[i]))

    predictions = []
    for i in range(len(preds)):
        predictions.append(int(preds[i]))

    acc = accuracy_score(trues, predictions)
    prec = precision_score(trues, predictions, average='macro')
    recall = recall_score(trues, predictions, average='macro')
    f1 = f1_score(trues, predictions, average='macro')


    data = []
    with open('Results.csv','r') as file:
        reader = csv.reader(file,dialect='excel')
        for i in reader:
            data.append(i)

    newline = [modelname+tvfile,float(acc), float(prec), float(recall), float(f1)]
    data.append(newline)
    with open('Results.csv', 'w', newline='') as file:
        writer = csv.writer(file,dialect='excel')
        writer.writerows(data)
        #writer.writerows(map(lambda x: [x], newline))
        file.close()

if __name__ == '__main__':
    with open('Results.csv','w',newline='') as csvfile:
        pass
    
    cr(tvfile='L', modelname='MRD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',modelfolder='Best_Models',dim=(149, 25) )
    cr(tvfile='CST', modelname='CSD',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\MOS_Images',data_labels='ZPLabels',modelfolder='Best_Models',dim=(149, 25) )

    cr(tvfile='L', modelname='MRLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',modelfolder='Best_Models',dim=(199, 25) )
    cr(tvfile='CST', modelname='CSLSTM',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Norm_zp_Ver',data_labels='ZPLabels',modelfolder='Best_Models',dim=(199, 25) )

    cr(tvfile='L', modelname='MRLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',modelfolder='Best_Models',dim=(199, 50) )
    cr(tvfile='CST', modelname='CSLSTM2P',actions=60,dataset ='F:\Work\Action recognition NTU\Datasets\Mos_Ver_2P',data_labels='ZPLabels',modelfolder='Best_Models',dim=(199, 50) )