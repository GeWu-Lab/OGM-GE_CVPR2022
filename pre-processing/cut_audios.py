# -*- coding: UTF8 -*-
import numpy as np
import librosa
import pickle
import os

import pdb
import csv
from moviepy.editor import *

def get_FileSize(filePath):
    filePath = str(filePath)
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)
    return round(fsize,2)

def audio_extract(audio_path,save_path):

    print(audio_path)
    video = VideoFileClip(audio_path)
    audio = video.audio
    audio.write_audiofile(save_path)

    print('done')



audio_dir = source
save_dir = target

audios = []

with open('vggsound.csv','r') as fid:
    csv_reader = csv.reader(fid)
    for item in csv_reader:
        #print(video_dir + '/' + item[0])
        if os.path.exists(audio_dir + '/' + item[0] + '_' + item[1] +'.mp4'):
            audios.append(item[0] + '_' + item[1] +'.mp4')

print(len(audios))


vid_count = 0

for each_audio in audios:
    if not each_audio.endswith('.mp4'):
        continue
    if each_audio == '20sstRN_pmI_170.mp4':
        continue
    audio_path = os.path.join(audio_dir, each_audio)
    save_path = os.path.join(save_dir, each_audio[:-4] + '.wav')

    if os.path.exists(save_path):
        continue
    if get_FileSize(audio_path) < 1.0:
        continue
    audio_extract(audio_path,save_path)
    #pdb.set_trace()
print('cut %d audio' % vid_count)