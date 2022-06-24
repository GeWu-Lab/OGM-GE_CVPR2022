### Pre-processing for different datasets:

VGGSound & Kinetics-Sound:

In the file VGG&KS we provide the pre-processing code, which can process the data into following dir tree:
```
VGG&KS
│------ train
│---------sample1
│------------frame1
│------------frame2
│------------sample1.wav
│------ test
│---------sample1
│------------frame1
│------------frame2
│------------sample1.wav
```


AVE&CREMA-D:

In the file AVE&CREMAD we provide the pre-processing code, which can process the data into following dir tree:
```
AVE/CREMA-D
│------ visual
│---------sample1
│------------frame1
│------------frame2
│------ audio
│---------sample1.wav
│---------sample2.wav
```

You can use video_to_frame.py and video_to_wav.py to process data. (we also provide old version preprocess_audio.py and video2frame-1fps.py, which you cna refer to.) To be noticed that .flv files can be processed similarly like mp4 files using VideoFileClip. And the code to process frames we provide is able to deal structure divided by classes, you can also ignore that by add a new middle level dir. 

