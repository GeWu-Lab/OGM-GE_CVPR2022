import copy
import csv
import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AVDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode

        if args.dataset == 'VGGSound':
            pass
        elif args.dataset == 'KineticSound':
            pass
        elif args.dataset == 'CREMAD':
            pass
        elif args.dataset == 'AVE':
            self.visual_feature_path = '/home/yake_wei/data/AVE_av/visual/'
            self.audio_feature_path = '/home/yake_wei/data/AVE_av/audio_spec/'
            with open('/home/yake_wei/code/AVE_CVPR22/datasets/stat.txt') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    classes.append(row[0])
            if mode == 'train':
                csv_file = '/home/yake_wei/code/AVE_CVPR22/datasets/my_train.txt'
            else:
                csv_file = '/home/yake_wei/code/AVE_CVPR22/datasets/my_test.txt'
            with open(csv_file) as f:
                csv_reader = csv.reader(f)
                for item in csv_reader:
                    audio_path = os.path.join('/home/yake_wei/data/AVE_av/audio_spec', item[1] + '.pkl')
                    visual_path = os.path.join('/home/yake_wei/data/AVE_av/visual', item[1])
                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        a = set(data)
                        if item[1] in a:
                            del data2class[item[1]]
                            data.remove(item[1])
                        else:
                            data.append(item[1])
                            data2class[item[1]] = item[0]
                    else:
                        continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.pkl')
        spectrogram = pickle.load(open(audio_path, 'rb'))

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        file_num = len(os.listdir(visual_path))

        if self.mode == 'train':

            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            t[i] = seg * i + 1
            path1.append('frame_0000' + str(t[i]) + '.jpg')
            image.append(Image.open(visual_path + "/" + path1[i]).convert('RGB'))
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file
