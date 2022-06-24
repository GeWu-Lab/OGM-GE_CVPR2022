import os
import logging
import time
import sys
import multiprocessing
from imageio import imsave
from moviepy.editor import VideoFileClip, concatenate_videoclips
import warnings
import cv2

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
        '--frame_output',
        default='/home/user/data/AVE_av/audio/',
        type=str,
        help='Output directory path of videos')
    return parser.parse_args()


def mp4_to_frame(dataset, dir_path,dst_dir_path):
    '''
    this function is to convert raw mp4 file into jpg frames.
    :param dataset: dataset, four choices by default.
    :param dir_path: the raw data dir.
    :param dst_dir_path: the processed data dir.
    :return:
    '''
    if dataset == 'AVE':
        pass
    elif dataset == 'KS':
        pass
    elif dataset == 'CREMAD':
        pass
    elif dataset == 'VGGSound':
        pass

    class_list = os.listdir(dir_path)
    class_list.sort()
    #print('--------------class_list :', class_list, '-----------\n')

    cpu_count = multiprocessing.cpu_count()
    process_count = cpu_count * 2 - 1
    pool = multiprocessing.Pool(process_count)
    deal_dir(dir_path, dst_dir_path, pool)

def log_config(logger=None, file_name="log"):
    '''
    this function is to save processing logs.
    :param logger: if new logger is needed.
    :param file_name: file name ends with log.
    :return:
    '''
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not os.path.exists("log"):
        os.makedirs("log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if file_name is not None:
        fh = logging.FileHandler("log/%s-%s.log" % (file_name, time.strftime("%Y-%m-%d")), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def deal_dir(dir_path, dst_path, pool=None):
    '''
    this function is to process dirs with classes.
    :param dir_path: source dir.
    :param dst_path: destination dir.
    :param pool: multiprocessing.
    :return:
    '''
    class_list = os.listdir(dir_path)
    class_list.sort()
    class_list.reverse()
    logger.info("----- deal dir: (%s), to path: (%s) -----", dir_path, dst_path)
    request_param = []
    for index in range(len(class_list)):
        class_name = class_list[index]
        logger.info("start deal class dir(%s), index: %s", class_name, index + 1)
        dir_path_class = os.path.join(dir_path, class_name)
        dst_path_class = os.path.join(dst_path, class_name)
        # check path is valid
        if not os.path.isdir(dir_path_class):
            logger.warning("process path(%s) is not dir ", dir_path_class)
            continue
        if not os.path.exists(dst_path_class):
            os.makedirs(dst_path_class)

        param = {'dir_path_class': dir_path_class, 'dst_path_class': dst_path_class}
        if pool is None:
            process_dir_path_class(param)
        else:
            request_param.append(param)

    if pool is not None:
        pool.map(process_dir_path_class, request_param)
        pool.close()
        pool.join()


def deal_video(video_file, out_path):
    '''
    this function is to process a video into frames.
    :param video_file: the video file path.
    :param out_path: the output data path.
    :return:
    '''
    if os.path.exists(out_path):
        return
    total_temp = 1
    if total_temp == 0:
        return
    else:
        try:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if not os.path.isfile(video_file):
                logger.error("deal video error, %s is not a file", video_file)
                return
            with VideoFileClip(video_file) as clip:
                video_len_pre = 10
                step = 1
                reader = clip.reader
                fps = clip.reader.fps
                total_frames = reader.nframes
                last_frames = int(total_frames % fps)
                if last_frames == 0:
                    last_frames = int(fps)
                last_start = total_frames - last_frames
                save_frame_index_arr = []


                for i in range(video_len_pre):
                    absolute_frame_pos = round((step / 2 + i) * fps)
                    if absolute_frame_pos > total_frames:
                        relative_frame_pos = last_start + 1 + ((absolute_frame_pos - last_start - 1) % last_frames)
                    else:
                        relative_frame_pos = absolute_frame_pos
                    print("--------relative_frame_pos: ",relative_frame_pos,'---------------')
                    save_frame_index_arr.append(relative_frame_pos)
                save_frame_map = {}
                loop_arr = list(set(save_frame_index_arr))
                loop_arr.sort()
                for i in loop_arr:
                    if i not in save_frame_map:
                        im = read_frame(reader, i)
                        save_frame_map[i] = im

                success_frame_count = 0
                for i in range(len(save_frame_index_arr)):
                    try:
                        out_file_name = os.path.join(out_path, "frame_{:05d}.jpg".format(i + 1))
                        im = save_frame_map[save_frame_index_arr[i]]
                        imsave(out_file_name, im)
                        success_frame_count += 1
                    except Exception as e:
                        logger.error("(%s) save frame(%s) error", video_file, str(i + 1), e)
                log_str = "video(%s) save frame, save count(%s) total(%s) fps(%s) %s, "
                if success_frame_count == video_len_pre:
                    logger.debug(log_str, video_file, success_frame_count, total_frames, fps, save_frame_index_arr)
                else:
                    logger.error(log_str, video_file, success_frame_count, total_frames, fps, save_frame_index_arr)

        except Exception as e:
            logger.error("deal video(%s) error", video_file, e)


def is_generate(out_path, dst=10):
    '''
    this function is to generate output dir.
    :param out_path: output path
    :param dst: target numbers
    :return:
    '''
    if not os.path.exists(out_path):
        return False
    folder_list = os.listdir(out_path)
    jpg_number = 0
    for file_name in folder_list:
        if file_name.strip().lower().endswith('.jpg'):
            jpg_number += 1
    return jpg_number >= dst


def fixed_video(clip, video_len_pre):
    '''
    this function is to deside the clip to cut off.
    :param clip: clip we want to get
    :param video_len_pre: the length of video
    :return:
    '''
    if clip.duration >= video_len_pre:
        return clip
    t_start = int(clip.duration)
    if t_start == clip.duration:
        t_start = -1
    last_clip = clip.subclip(t_start)
    final_clip = clip
    while final_clip.duration < video_len_pre:
        final_clip = concatenate_videoclips([final_clip, last_clip])
    return final_clip

def read_frame(reader, pos):
    '''
    this function is to read frames.
    :param reader: reader unit
    :param pos: current position
    :return:
    '''
    if not reader.proc:
        reader.initialize()
        reader.pos = pos
        reader.lastread = reader.read_frame()

    if pos == reader.pos:
        return reader.lastread
    elif (pos < reader.pos) or (pos > reader.pos + 100):
        reader.initialize()
        reader.pos = pos
    else:
        reader.skip_frames(pos - reader.pos - 1)
    result = reader.read_frame()
    reader.pos = pos
    return result

def compute_numbers(outpath):
    '''
    this function is to compute the numbers of .jpg file in a dir.
    :param outpath: the dir to count.
    '''
    folder_list = os.listdir(outpath)
    folder_list.sort()
    jpg_number = 0
    for file_name in folder_list:
        if '.jpg' in file_name:
            jpg_number += 1
    if jpg_number < 6:
        print('-----------jpg_number: ',jpg_number,'-------------------\n')
        return 1
    else:
        return 0

logger = log_config()
if __name__=="__main__":
    args = get_arguments()
    dataset, dir_path, dst_dir_path = args.dataset, args.video_input, args.frame_output
    print('start converting for dataset' + dataset)
    mp4_to_frame(dataset, dir_path,dst_dir_path)
    print('finish!')