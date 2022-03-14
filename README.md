# OGM-GE implemented in PyTorch
**Paper's Name: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"
Accepted by: IEEE Conference on Computer Vision and Pattern Recognition(2022), ID 6810.
The link of ready paper will be released soon**


## Method Introduction
Here is the code example of OGM-GE training strategy, which is a easy way to apply to different vanilla audio_visual fusion networks or other audio_visual networks, helping to achieve a more balanced and adequate training. More analysis and details can be seen in the paper.

This code includes four dirs: 
data to store data of database; 
dataloader that including three dataloaders(audi_visual,audio,visual); 
models that including vanilla audio_visual models; 
tempresult to save more temporary result, 
and we also provid training, testing, and tools to evaluate more performance index.

![Method Pipeline](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/blob/main/demo/pipeline.pdf)


## Main Training Environment
+ Ubuntu 16.04
+ CUDA Version: 11.1
+ PyTorch 1.8.1
+ torchvision 0.9.1
+ python 3.7.6


## Data Preparation
### Original data
Original Dataset：[AVE](https://sites.google.com/view/audiovisualresearch)
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset)
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
[ESC50](https://github.com/karoldvl/ESC-50/archive/master.zip)



### Pre-processing

For example, we provide code to pre-process videos into RGB frames and audio wav files. By running python files with ```'python preprocess_audio.py'``` and ```'python video2frame-1fps.py'``` (they can also process raw data into other settings). The only thing you need to change is the data or other file addresses.


After downloading and processing data, you should build the data dir  following proper structure. We give an example of AVE dataset:
```
AVE
│------ visual
│------ audio
│--- other files
```

&nbsp;



### Train the model

Before training, you should new a directory named **`model`** to save checkpoint file. 
```python train_amd.py```
You can change related settings in file ```train_amd.py```.
&nbsp;


### Test and Eval

You can test the performance of trained model by simply running
```python test.py```
And you can also learn more information(mAP, and so on) by running
```python eval.py```


### Contact us

If you have any detailed questions or suggestions, you can email us:
**xiaokangpeng@ruc.edu.cn**
