import glob
import multiprocessing
import subprocess
import os
import argparse
import pdb

def get_FileSize(filePath):
    filePath = str(filePath)
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024)
    return round(fsize,2)

def get_arguments():
    parser = argparse.ArgumentParser()
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

def convert(v):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    args.audio_output + '%s.wav' % v.split('/')[-1][:-4]])

def obtain_list():
    files = []
    already = 0
    txt = glob.glob(args.video_input + '/*.mp4')
    for item in txt:
        if os.path.exists(args.audio_output + item.split('/')[-1][:-4] + '.wav'): #or get_FileSize(item) < 10.0:
            already += 1
        else:
            files.append(item)
    print(len(files))
    pdb.set_trace()
    return files

args = get_arguments()
p = multiprocessing.Pool(512)
p.map(convert, obtain_list())

