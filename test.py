import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import csv
from models import AVmodel
from datasets import AVDataset
import warnings
import pdb
warnings.filterwarnings('ignore')



# basic settings, CREMA-D for instance
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
        default='test.csv',
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



def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AVmodel(args)
    model.to(device)
    #model.load_state_dict(torch.load('trainmodel.pth'))   # load your model here
    model.to(device)
    print('load pretrained model.')


    testdataset = AVDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 16)#num_workers = 16

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")

    with torch.no_grad():

        model.eval()
        num = [0.0 for i in range(6)]
        acc = [0.0 for i in range(6)]
        racc = [0.0 for i in range(6)]

        for step, (spec, image, label, name) in enumerate(testdataloader):

            spec = Variable(spec).cuda()
            image = Variable(image).cuda()
            label = Variable(label).cuda()

            out = model(spec.unsqueeze(1).float(), image.float(),label,-1)
            prediction = softmax(out)


            for i, item in enumerate(name):
                np.save(args.result_path + '/%s.npy' % item, prediction[i].cpu().data.numpy())


                ma=np.max(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0
                #print(ma,prediction[i].cpu().data.numpy()[label[i]])
                if (abs(prediction[i].cpu().data.numpy()[label[i]]-ma) <= 0.0001):
                   # print('match')
                    acc[label[i]] += 1.0
                #print(classes[label[torch.argmax(prediction[i].cpu().data.numpy())]])

        for i in range(0,6):
            print('class label:',i,'sum:',num[i],'acc:',acc[i])
            if num[i] != 0.0:
                racc[i] = acc[i]/num[i]
                if(acc[i]==0.0):
                    racc[i]=0.0001
            print('racc',racc[i])

    print('test acc:',sum(acc)/sum(num))
    print('test macc:',sum(racc)/6)



if __name__ == "__main__":
    main()

