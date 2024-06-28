import csv
import os

import numpy as np
import torchaudio
import torch

## save path of processed spectrogram
save_path = 'train_spec'

## file path of wav files
audio_path='train_wav/train'

## the list of all wav files
csv_file = 'ks_train_real.txt'


data = []
with open(csv_file) as f:
  for line in f:
      item = line.split("\n")[0].split(" ")
      name = item[0][:-4]

      if os.path.exists(audio_path + '/' + name + '.wav'):
        data.append(name)
        # print(name)
        # exit(0)

for name in data:
  waveform, sr = torchaudio.load(audio_path + '/'+ name + '.wav')
  waveform = waveform - waveform.mean()
  norm_mean = -4.503877
  norm_std = 5.141276

  fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
  
  target_length = 1024
  n_frames = fbank.shape[0]
  # print(n_frames)
  p = target_length - n_frames

  # cut and pad
  if p > 0:
      m = torch.nn.ZeroPad2d((0, 0, 0, p))
      fbank = m(fbank)
  elif p < 0:
      fbank = fbank[0:target_length, :]
  fbank = (fbank - norm_mean) / (norm_std * 2)

  print(fbank.shape)
  np.save(save_path + '/'+ name + '.npy',fbank)
