import os
import cv2
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import matplotlib.pyplot as plt
import glob
import sys
from scipy import signal
import random
import soundfile as sf
import librosa
import pdb
import copy
import math
import pandas as pd

num_class = 309
ratio = 100
max = 1000
min = max/ratio
start = math.log(max)
end = math.log(min)

print(start,end)

x = [0.0] * num_class
y = [0] * num_class
#pick = [0] * num_class
classes = []
data = []
data1 = []
dict = {}
data2class = {}
pick2class = {}
num2class = {}

for i in range(num_class):
    x[i] = start - i*(start - end)/num_class
    y[i] = int(math.exp(x[i]))
    #dict[classes[i]]=0





with open('stat.csv') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        classes.append(row[0])

print(len(classes))

for i in range(num_class):

    pick2class[classes[i]] = 0
    num2class[classes[i]] = y[i]

'''
with open('stat_pick.csv','r') as csvFile:
    rows = csv.reader(csvFile)
    with open('stat_pick2.csv','w',newline='') as f:
        writer = csv.writer(f)
        i = 0
        for row in rows:
            row.append(y[i])
            print(i)
            i = i + 1
            writer.writerow(row)
'''



with open('train_all.csv') as f:
    csv_reader = csv.reader(f)
    for item in csv_reader:
        #print(item)
        if item[2] in classes and os.path.exists(audio_path)and os.path.exists(visual_path) and pick2class[item[2]]<num2class[item[2]]: #and pick2class[classes]<=100:


            data.append(item[0])
            data1.append(item[1])
            data2class[item[0] + '_' + item[1]] = item[2]
            pick2class[item[2]] += 1

with open('train_lt.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(data)):
        row = []
        row.append(data[i])
        row.append(data1[i])
        row.append(data2class[data[i]+'_'+data1[i]])
        #print(i)

        writer.writerow(row)


print('finish')



