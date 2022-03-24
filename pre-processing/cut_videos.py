import os
import cv2
import pdb
import csv

def video2frame(video_path, frame_save_path, frame_interval=1):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    #pdb.set_trace()
    success, image = vid.read()
    count = 0
    while success:
        count +=1
        if count % frame_interval == 0:
            #cv2.imencode('.png', image)[1].tofile(frame_save_path+'/fame_%d.png'%count)
            save_name = '{}/frame_{}_{}.jpg'.format(frame_save_path, int(count/fps),count)
            cv2.imencode('.jpg', image)[1].tofile(save_name)
        success, image = vid.read()
    print(count)


def video2frame_update(video_path, frame_save_path, frame_kept_per_second=1):

    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)
    video_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        video_len = int(video_frames/fps)
    except:
        return
    #print(fps)
    #print(video_len)

    count = 0
    frame_interval = int(fps/frame_kept_per_second)
    while(count < fps*video_len):
        ret, image = vid.read()
        if not ret:
            break
        if count % fps == 0:
            frame_id = 0
        if frame_id<frame_interval*frame_kept_per_second and frame_id%frame_interval == 0:
            save_dir = '{}/frame_{:03d}'.format(frame_save_path, int(count/fps))
            #pdb.set_trace()
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            save_name = '{}/frame_{:03d}/{:05d}.jpg'.format(frame_save_path, int(count/fps), count)
            cv2.imencode('.jpg', image)[1].tofile(save_name)
        
        frame_id += 1
        count += 1


video_dir = ' '
videos = []
with open('vggsound.csv','r') as fid:
    csv_reader = csv.reader(fid)
    for item in csv_reader:
        #print(video_dir + '/' + item[0])
        if os.path.exists(video_dir + '/' + item[0] + '_' + item[1] +'.mp4'):
            videos.append(item[0] + '_' + item[1] +'.mp4')
    #videos = [line.strip().split(' ')[1] for line in fid.readlines()]
save_dir = ' '

print(len(videos))

pdb.set_trace()
vid_count = 0

for each_video in videos:
    if not each_video.endswith('.mp4'):
        continue
    print(each_video)
    video_path = os.path.join(video_dir, each_video)
    save_path = os.path.join(save_dir, each_video[:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        continue
    video2frame_update(video_path, save_path, frame_kept_per_second=1)
    #pdb.set_trace()
print('cut %d videos' % vid_count)
