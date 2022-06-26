import cv2
import csv
import math
from PIL import Image
from matplotlib import cm
import numpy as np
import time
import os
import sys

def settozero(data):
    x = data[0][3]
    y = data[0][4]
    z = data[0][5]

    #print(x,y,z)

    diffx = -float(x)
    diffy = -float(y)
    diffz = 2.5-float(z)
    #print(diffx,diffy,diffz)

    for i in range(len(data)):
        for j in range(25):
            data[i][j*3] = float(data[i][j*3]) + diffx
            data[i][j*3+1] = float(data[i][j*3+1]) +  diffy
            data[i][j*3+2] = float(data[i][j*3+2]) +  diffz
    return data

def loadcsv(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    return data
def savecsv(path, data):
    with open(path, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

'''
-- arg[0] : DataPath for Skeleton Csv Files
-- arg[1] : New Path to save new Csv Files
'''
dataset = sys.argv[0]
savepath = sys.argv[1]
if not os.path.exists(savepath):
    os.makedirs(savepath)

if __name__ == '__main__':

    for filename in os.listdir(dataset):
        if filename.endswith(".csv"):
            print(filename)
            data = loadcsv(dataset+filename)
            data = settozero(data)
            savecsv(savepath+filename, data)