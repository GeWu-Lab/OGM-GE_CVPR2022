import csv
import os
import glob
import sys
import numpy as np
from utils import *
import argparse
import pdb

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
        '--data_stat',
        default='./data/stat.csv',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--data_path',
        default='./data/test_all.csv',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--result_path',
        default='result_path',
        type=str,
        help='metadata directory')
    return parser.parse_args() 

def main():
    args = get_arguments()
    classes = []
    data = []
    data2class = {}

    # load classes
    with open(args.data_stat) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            classes.append(row[0])
    classes = sorted(classes)
    print(classes)
    # load test data
    with open(args.data_path) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[2] in classes and os.path.exists(
                        args.audio_path + '/' + row[0] + '_' + row[1] + '.wav') and os.path.exists(
                        args.visual_path + '/' + row[0] + '_' + row[1]) and os.path.exists(args.result_path +'/'+ row[0] + '_' + row[1] + '.npy'):
                data2class[row[0] + '_' + row[1]] = row[2]
                data.append(row[0] + '_' + row[1])

    # placeholder for prediction and gt
    print(len(data))
    pred_array = np.zeros([len(data),len(classes)])
    gt_array = np.zeros([len(data),len(classes)])


    for count, item in enumerate(data):

        #pdb.set_trace()
        pred = np.load(args.result_path +'/'+ item + '.npy')

        label = data2class[item]
        label_index = []
        label_index.append(classes.index(label))
        
        pred_array[count,:] = pred
        gt_array[count,np.array(label_index)] = 1


    stats = calculate_stats(pred_array,gt_array)
    count = 0
    mAP = 0.0
    aps = [0]*309
    for stat in stats:
        if str(stat['AP']) != 'nan':
            aps[count] = stat['AP']
            count += 1
            print(stat['AP'])
            mAP += stat['AP']
    mAP /= count
    #mAP = np.mean([stat['AP'] for stat in stats])
    #mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))
    #print("mAUC: {:.6f}".format(mAUC))
    #print("dprime: {:.6f}".format(d_prime(mAUC)))

if __name__ == "__main__":
    main()

