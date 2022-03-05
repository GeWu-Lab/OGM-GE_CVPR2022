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
import glob
import sys
from scipy import signal
import random
import soundfile as sf
import librosa
import pdb

class GetAudioVideoDataset(Dataset):

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
        #print('stat.csv load over')
        #print(classes)
        with open(args.csv_path  + args.test) as f:

            csv_reader = csv.reader(f)
          
            for item in csv_reader:
                
                #if i > 300:
                    #break
                #print(args.data_path + item[0][:11] + '.flac')
                #print(item[1])
                if item[3] in classes and os.path.exists(args.data_path + item[0] + '.flac'):
                    #print('pass')
                    print(args.data_path + item[0] + '.flac')
                    data.append(item[0])
                    data2class[item[0]] = item[3]
        print(classes)
        #print(args.csv_path + args.test)
        print('data load over')
        print(len(data))
        self.audio_path = args.data_path 
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
        wav_file = self.video_files[idx]
        # Audio
        #samples, samplerate = sf.read(self.audio_path + wav_file+'.flac')
        samples, samplerate = librosa.load(self.audio_path + wav_file+'.flac')

        # repeat in case audio is too short
        resamples = np.tile(samples,10)[:160000]
        #pdb.set_trace()

        '''
        wav = resamples
        sr = 16000
        secs = int(len(wav) / sr)
        #print(secs)
        for i in range(secs):
            start = sr * i
            end = sr * (i + 1)
            cur_wav = wav[start:end]
            break
        resamples = cur_wav
        '''



        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)

        return spectrogram, resamples,self.classes.index(self.data2class[wav_file]),wav_file


