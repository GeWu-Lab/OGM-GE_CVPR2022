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




class AVmodel_uni(nn.Module):
    def __init__(self,args):
        super(AVmodel_uni,self).__init__()
        self.args = args
        self.parta = Aencoder(self.args)
        self.partv = Vencoder(self.args)
        
        
        self.fc_ = nn.Linear(1024, args.n_classes)

        self.fc_a = nn.Linear(512, args.n_classes)
        self.fc_v = nn.Linear(512, args.n_classes)


    def forward(self,audio,visual,iterations):

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

        out_y = self.fc_a(y)
        out_x = self.fc_v(x)


        return out_x, out_y

