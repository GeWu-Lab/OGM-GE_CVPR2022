import os


train_videos = '/data/users/xiaokang_peng/VGGsound/train-videos/train_video_list.txt'
test_videos = '/data/users/xiaokang_peng/VGGsound/test-videos/test_video_list.txt'

train_audio_dir = '/data/users/xiaokang_peng/VGGsound/train-audios/train-set'
test_audio_dir = '/data/users/xiaokang_peng/VGGsound/test-audios/test-set'


# test set processing
with open(test_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    mp4_filename = os.path.join('/data/users/xiaokang_peng/VGGsound/test-videos/test-set/', item[:-1])
    wav_filename = os.path.join(test_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))


# train set processing
with open(train_videos, 'r') as f:
    files = f.readlines()

for i, item in enumerate(files):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(files)))
        print('*******************************************')
    mp4_filename = os.path.join('/data/users/xiaokang_peng/VGGsound/train-videos/train-set/', item[:-1])
    wav_filename = os.path.join(train_audio_dir, item[:-5]+'.wav')
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))





