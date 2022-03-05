import os
import sys
from PIL import Image
import torch
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse
import csv
import random
import warnings
import pdb
from encodera import Aencoder
from encoderv import Vencoder
warnings.filterwarnings('ignore')


# basic concatenation framework for audio-visual learning
class AVmodel(nn.Module):
    def __init__(self,args):
        super(AVmodel,self).__init__()
        self.args = args
        self.parta = Aencoder(self.args)
        self.partv = Vencoder(self.args)

        self.relu = nn.ReLU(inplace=True)
        self.fc_ = nn.Linear(1024, 309)



    def forward(self,audio,visual,label,iterations):
        
        iteration = iterations
        y = self.parta(audio)
        x = self.partv(visual)
         

        (_, C, H, W) = x.size()
        B = y.size()[0]
        x = x.view(B, -1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)


        x = F.adaptive_avg_pool3d(x, 1)
        y = F.adaptive_avg_pool2d(y, 1)
        x = x.squeeze(2).squeeze(2).squeeze(2)
        y = y.squeeze(2).squeeze(2)
        out = torch.cat((x, y),1)
        out = self.fc_(out)



        return x,y, out

