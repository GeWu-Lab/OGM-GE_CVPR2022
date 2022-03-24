import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import random
import argparse
import csv
import math
from models.avmodel import AVmodel     # audio-visual model as example
from datasets import AVDataset
import warnings
import pdb
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/train_normal')


# Basic settings
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--audio_path',
        default='audio_path',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--visual_path',
        default='visual_path',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--result_path',
        default='result_path',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--summaries',
        default='summaries',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--csv_path',
        default='./data/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--test',
        default='train.csv',
        type=str,
        help='test csv files')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=1000,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    return parser.parse_args()


# Setup the random seed
def setup_seed(seed=0):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     os.environ['PYTHONHASHSEED'] = str(seed)


# Training function for 'Concatenation' fused model
def train(epoch, l, model, dataloader):

    lr_start = 0.001
    lr = lr_start * (0.1 ** (epoch // 70))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    print("start training")
    model.train()
    total_loss = 0.0

    for step, (spec, image, label, name) in enumerate(dataloader):

        iteration = epoch*l+step
        spec = Variable(spec).cuda()
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        optimizer.zero_grad()

        # Out_a, out_v are calculated to estimate the performance of 'a' and 'v' modality.
        x, y, out = model(spec.unsqueeze(1).float(), image.float(), label, iteration)
        out_v = (torch.mm(x,torch.transpose(model.module.fc_.weight[:,:512],0,1)) + model.module.fc_.bias/2)
        out_a = (torch.mm(y,torch.transpose(model.module.fc_.weight[:,512:],0,1)) + model.module.fc_.bias/2)
        loss = criterion(out, label)

        # Calculate original loss first.
        loss.backward()
        # Calculation of discrepancy ration and k.
        k_a,k_v = calculate_coefficient(label, out_a, out_v)
        # Gradient Modulation begins before optimization, and with GE applied.
        update_model_with_OGM_GE(model, k_a, k_v)
        # Optimize the modulated parameters.
        optimizer.step()
        total_loss += loss.item()

    total_loss = total_loss/len(dataloader)
    return total_loss


def calculate_coefficient(label, out_a, out_v):
        softmax = nn.Softmax(dim=1)
        relu = nn.ReLU(inplace=True)
        ratio = sum([softmax(out_a)[i][label[i]] for i in range(out_a.size(0))])/sum([softmax(out_v)[i][label[i]] for i in range(out_v.size(0))])
        writer.add_scalar('data/ratio',ratio,iteration)

        k_v = relu(1/ratio-1)
        k_v = torch.tanh(k_v)
        writer.add_scalar('data/k_v',k_v,iteration)

        # Here, 0.5 is the hyper-parameter that can be adjusted according to the data.
        k_a = relu(0.5*ratio)
        k_a = torch.tanh(k_a)
        writer.add_scalar('data/k_a',k_a,iteration)
        return k_a,k_v

def update_model_with_OGM_GE(model, k_a, k_v):
    for name, parms in model.named_parameters(): 
        layer=str(name).split('.')[1]

        if layer=='parta' and len(parms.grad.size())==4:
            parms.grad *= (1 - k_a)
            parms.grad += torch.zeros_like(parms.grad).normal_(0,parms.grad.std().item())

        if layer=='partv' and len(parms.grad.size())==4:
            parms.grad *= (1 - k_v)
            parms.grad += torch.zeros_like(parms.grad).normal_(0,parms.grad.std().item())

#Valid function
def valid(model, dataloader):

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        model.eval()
        num = [0.0 for i in range(6)]
        acc = [0.0 for i in range(6)]
        acc_a = [0.0 for i in range(6)]
        acc_v = [0.0 for i in range(6)]

        for step, (spec, image, label, name) in enumerate(dataloader):

            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            label = Variable(label).cuda()
            x,y,out = model(spec.unsqueeze(1).float(), image.float(),label,-1)
            out_v = (torch.mm(x,torch.transpose(model.module.fc_.weight[:,:512],0,1)) + model.module.fc_.bias/2)
            out_a = (torch.mm(y,torch.transpose(model.module.fc_.weight[:,512:],0,1)) + model.module.fc_.bias/2)
            prediction = softmax(out)

            # We record acc and acc for uni-modals separately.
            for i, item in enumerate(name):

                ma = np.max(prediction[i].cpu().data.numpy())
                v = np.max(out_v[i].cpu().data.numpy())
                a = np.max(out_a[i].cpu().data.numpy())
                num[label[i]] += 1.0
                if (abs(prediction[i].cpu().data.numpy()[label[i]] - ma) <= 0.0001):
                    acc[label[i]] += 1.0
                if (abs(out_v[i].cpu().data.numpy()[label[i]] - v) <= 0.0001):
                    acc_v[label[i]] += 1.0
                if (abs(out_a[i].cpu().data.numpy()[label[i]] - a) <= 0.0001):
                    acc_a[label[i]] += 1.0

    return sum(acc_v)/sum(num),sum(acc_a)/sum(num),sum(acc)/sum(num)


def main():

    args = get_arguments()
    setup_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model = AVmodel(args)
    model.cuda()

    device_ids = [0,1,2,3]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    traindataset = AVDataset(args, mode='train')
    traindataloader = DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,num_workers=32, pin_memory = True)

    args.test = 'valid.csv'
    testdataset = AVDataset(args, mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=32)


    epoch = 100
    best_acc = 0.0
    print("Loaded dataloader.")

    # Training and valid
    for e in range(epoch):
        print('Epoch %d' % e)
        loss = train(e, len(traindataloader), model,traindataloader)
        acc_v,acc_a,acc = valid(model,testdataloader)
        writer.add_scalar('data/loss',loss,e)
        writer.add_scalar('data/acc',acc,e)
        writer.add_scalar('data/acc_a',acc_a,e)
        writer.add_scalar('data/acc_v',acc_v,e)

        if acc > best_acc:
            best_acc = float(acc)
            torch.save(model.state_dict(), './trainmodel_OGM_GE.pth')
            print("loss:%03f,acc:%03f, model saved" % (loss,acc))
        else:
            print("loss:%03f,acc:%03f" % (loss,acc))


if __name__ == "__main__":
    main()
