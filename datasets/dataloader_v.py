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

class GetAudioVideoDataset_v(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data2path = {}
        classes = []
        classes_ = []
        data = []
        data2class = {}

        with open(args.csv_path + 'stat.csv') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        with open(args.csv_path + args.test) as f:

            csv_reader = csv.reader(f)
            for item in csv_reader:

                #print(args.data_path + '/'+ item[0])
                #print(item[3])
                if os.path.exists(args.data_path + '/'+ item[0]) and os.path.exists(args.data_path + '/'+ item[0]+'/'+'frame_000'):
                    print(args.data_path + '/' + item[0])
                    if item[0] in data:
                        continue
                    data.append(item[0])
                    data2class[item[0]] = item[3]
        #print(classes)
        #print(args.csv_path + args.test)
        print('data load over')
        print(len(data))
        self.video_path = args.data_path
        self.mode = mode
        self.transforms = transforms
        self.classes = sorted(classes)
        self.data2class = data2class

        # initialize audio transform
        self._init_atransform()
        #  Retrieve list of audio and video files
        self.video_files = []

        for item in data:
            self.video_files.append(item)
        print('# of audio files = %d ' % len(self.video_files))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        image_file = self.video_files[idx]
        path = self.video_path + '/' + image_file
        file_num = len([lists for lists in os.listdir(path) if os.path.isdir(os.path.join(path, lists))])
        t=random.randint(0, file_num-1)
        i=0

        for lists in os.listdir(path):
            if i==t:
                path2 = lists
                break
            i += 1

        path1=""
        file_dir = path + "/" +path2
        for root, dirs, files in os.walk(file_dir):
            if files == []:
                continue
            path1 = files[0]
            break



        image = Image.open(file_dir + "/" + path1).convert('RGB')

        transf = transforms.ToTensor()
        trans2 = transforms.Resize(size=(224, 224))
        image = trans2(image)
        image_arr = transf(image)
        #image_arr.resize_(3,224,224)
        #print(image_arr.size())




        return image_arr, self.classes.index(self.data2class[image_file]),image_file
                                                                                            