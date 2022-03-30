import multiprocessing
import os
import os.path
import pickle

import librosa
import numpy as np
from scipy import signal


def audio_extract(path, audio_name, audio_path, sr=16000):
    save_path = path
    samples, samplerate = librosa.load(audio_path)
    resamples = np.tile(samples, 10)[:sr]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    assert spectrogram.shape == (257, 1004)
    save_name = os.path.join(save_path, audio_name + '.pkl')
    print(save_name)

    with open(save_name, 'wb') as fid:
        pickle.dump(spectrogram, fid)


class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('{}: Exiting'.format(proc_name))
                self.task_queue.task_done()
                break
            # print(next_task)
            audio_extract(next_task[0], next_task[1], next_task[2])
            self.task_queue.task_done()


if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count()
    print('Creating {} consumers'.format(num_consumers))
    consumers = [
        Consumer(tasks)
        for i in range(num_consumers)
    ]
    for w in consumers:
        w.start()

    # path='data/'
    save_dir = '/home/xiaokang_peng/data/AVE_av/audio_spec'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path_origin = '/home/xiaokang_peng/data/AVE_av/audio'
    audios = os.listdir(path_origin)
    for audio in audios:
        audio_name = audio
        audio_path = os.path.join(path_origin, audio)
        tasks.put([save_dir, audio_name[:-4], audio_path])

    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()

    print("ok")
