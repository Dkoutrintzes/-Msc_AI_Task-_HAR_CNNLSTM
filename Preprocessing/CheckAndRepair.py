import cv2
import csv
import math
from PIL import Image
from matplotlib import cm
import numpy as np
import time
import os
import sys
#NZ_R
'''
-- arg[0] : DataPath for Skeleton Csv Files
-- arg[1] : New Path to save new Csv Files
'''
dataset = sys.argv[0]
savepath = sys.argv[1]

if not os.path.exists(savepath):
    os.makedirs(savepath)

def loadcsv(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = list(reader)
    return data
def savecsv(path, data):
    with open(path, 'w',newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

def check_zerolines(data):
    counter = 0
    for line in data:
        
        #print(len(line))
        fp = line[:75]
        sp = line[75:]
        #print(fp)
        #print(len(fp),len(sp))
        zerocounter = 0
        for n in range(len(fp)):
            #print(n)
            if float(fp(n)) == 0.0 and float(sp(n)) == 0.0:
                zerocounter += 1
        if zerocounter > 9:
            counter += 1
    return counter

def check_mixedpersons(data):
    newdata = []
    for i in range(len(data)):
        if i == 0:
            newdata.append(data[i])
            continue
        #print(len(line))
        lfp = data[i-1][:75]
        fp = data[i][:75]
        sp = data[i][75:]
        #print(fp)
        if len(fp) == len(sp):
            counter = 0
            for i in range(len(fp)):
                #print(abs(float(fp[i]) - float(lfp[i])) , abs(float(sp[i]) - float(lfp[i])))
                if abs(float(fp[i]) - float(lfp[i])) > abs(float(sp[i]) - float(lfp[i])):
                        temp = fp[i]
                        fp[i] = sp[i]      
                        sp[i] = temp
        newdata.append(fp+sp)
                       
        
    return newdata  
    
                 
                
if __name__ == '__main__':

    for filename in os.listdir(dataset):
        #print(filename)
        if filename.endswith(".csv"):
            data = loadcsv(dataset+filename)
            emptylines = check_zerolines(data)
            if emptylines < 5:
                #print(np.shape(data))
                ndata = check_mixedpersons(data)
                #print(np.shape(ndata))
                savecsv(os.path.join(savepath,filename),ndata)
            
        
                
            