import glob
import multiprocessing
import subprocess
import os
import argparse
import pdb

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='AVE',
        type=str)
    parser.add_argument(
        '--video_input',
        default='/home/user/data/AVE/video',
        type=str,
        help='Input directory path of videos or audios')
    parser.add_argument(
        '--audio_output',
        default='/home/user/data/AVE_av/audio/',
        type=str,
        help='Output directory path of videos')
    return parser.parse_args()


def mp4_to_wave(dataset,dir_path,dst_dir_path):
    '''
    this function is to convert raw mp4 file into wav files.
    :param dataset: dataset, four choices by default.
    :param dir_path: the raw data dir.
    :param dst_dir_path: the processed data dir.
    '''
    if dataset == 'AVE':
        pass
    elif dataset == 'KS':
        pass
    elif dataset == 'CREMAD':
        pass
    elif dataset == 'VGGSound':
        pass
    p = multiprocessing.Pool(512)
    p.map(convert(dst_dir_path), obtain_list(dir_path))


def get_FileSize(filePath):
    '''
    this function is to get the size of a file.
    :param filePath: target file path.
    :return: size of the file(KB),float type.
    '''
    filePath = str(filePath)
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)
    return round(fsize,2)


def convert(v,dst_dir_path):
    '''
    this function is to translate mp4 into wav files quickly with multiprocessing.
    :param v: ffmpeg param.
    :param dst_dir_path: destination processed file dir.
    :return:
    '''
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    dst_dir_path + '%s.wav' % v.split('/')[-1][:-4]])


def obtain_list(dir_path):
    '''
    this function is to get the file list to process.
    return: list of mp4 files
    '''
    files = []
    already = 0
    txt = glob.glob(dir_path + '/*.mp4')
    for item in txt:
        if os.path.exists(dir_path + item.split('/')[-1][:-4] + '.wav'): #or get_FileSize(item) < 10.0:
            already += 1
        else:
            files.append(item)
    print(len(files))
    return files

if __name__=="__main__":
    args = get_arguments()
    dataset, dir_path, dst_dir_path = args.dataset, args.video_input, args.audio_output
    print('start converting for dataset' + dataset)
    mp4_to_wave(dataset,dir_path,dst_dir_path)
    print('finish!')