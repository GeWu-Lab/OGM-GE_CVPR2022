# OGM-GE implemented in PyTorch

Here is the source code of OGM-GE training strategy proposed in *Balanced Multimodal Learning via On-the-fly Gradient Modulation*, which is a easy way to apply to different vanilla multimodal fusion networks or other multimodal networks, helping to achieve a more balanced and adequate training. More analysis and details are displayed in the paper.

This code includes four main dirs: 
'data to store data of database; 
'dataloader' that including three dataloaders(audi_visual,audio,visual); 
'models' that including vanilla audio_visual models; 
'pre-processig' to pre-process video data, 
and we also provid training, testing, and tools.

*Paper Title: "Balanced Multimodal Learning via On-the-fly Gradient Modulation"*
*Authors: Xiaokang Peng*, Yake Wei*, Andong Deng, Dong Wang, and Di Hu*
*Accepted by: IEEE Conference on Computer Vision and Pattern Recognition(CVPR 2022)*
*The link of paper will be released soon*


## What's the imbalance phenomenon in multimodal learning task?
We observe that the potential of multimodal information is not fully exploited even when the multimodal model outperforms its uni-modal counterpart. We conduct linear probing experiments to explore the quality of jointly trained encoders, and find them under-optimized (the yellow line) compared with the uni-modal model (the red line). We proposed the OGM-GE method to improve the optimization process adaptively and achieved consistent improvement (the blue line). We improve both the multimodal performance and uni-model representation as shown in the following figure.
<div  align="center">    
<img src="demo/five lines.PNG" width = "80%" />
</div>

## Method Introduction
Pipeline of our OGM-GE method, consisting two steps:
1. On-the-fly gradient modulation for different modalities, which is designed to adaptively balance the training between modalities.
2. Adaptive Gaussion noise enhancement on gradient, which works as a versatile measure to boost model generalization ability.
<div  align="center">    
<img src="demo/pipeline.PNG" width = "80%" />
</div>


## Main Dependencies
+ Ubuntu 16.04
+ CUDA Version: 11.1
+ PyTorch 1.8.1
+ torchvision 0.9.1
+ python 3.7.6


## Data Preparation
### Original dataset
Original Dataset：
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset)
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
[AVE](https://sites.google.com/view/audiovisualresearch)
[ESC50](https://github.com/karoldvl/ESC-50/archive/master.zip)



### Pre-processing

For example, we provide code to pre-process videos into RGB streams at different frame rates and audio wav files in the file ```'tools'```. By running python files with ```'python pre-processing/preprocess_audio.py'``` and ```'python pre-processing/video2frame-1fps.py'```. The only thing you need to change is the data or other file addresses.You can also adjust the frame rates by changing the parameters in the ```'preprocess_audio.py'``` file, or the ```'video2frame-1fps.py'``` file.


After downloading and processing data, you should build the data dir  following proper structure. We give an example of AVE dataset:
```
AVE
│------ visual
│---------sample1
│------------frame1
│------------frame2
│------ audio
│---------sample1.wav
```

&nbsp;

## Core code demo

Our proposed OGM-GE can work as a simple but useful plugin for some widely used multimodal fusion frameworks. We dispaly the core code part as following:
```python
import torch

    ---in training step---
    
    # Out_a, out_v are calculated to estimate the performance of 'a' and 'v' modality.
    x, y, out = model(spec.unsqueeze(1).float(), image.float(), label, iteration)
    out_v = (torch.mm(x,torch.transpose(model.module.fc_.weight[:,:512],0,1)) + model.module.fc_.bias/2)
    out_a = (torch.mm(y,torch.transpose(model.module.fc_.weight[:,512:],0,1)) + model.module.fc_.bias/2)
    loss = criterion(out, label)

    # Calculate original loss first.
    loss.backward()
    
    # Calculation of discrepancy ration and k.
    k_a,k_v = calculate_coefficient(label, out_a, out_v)
    
    # Gradient Modulation begins before optimization, and with GE applied.
    update_model_with_OGM_GE(model, k_a, k_v)
    
    # Optimize the modulated parameters.
    optimizer.step()
    
    ---continue next training step---
```

## Train the model

Before training, you should new a directory named **`model`** to save checkpoint file. 
```python train_amd.py```
You can change related settings in file ```train_amd.py```.
&nbsp;


## Test and Eval

You can test the performance of trained model by simply running
```python test.py```
And you can also learn more information(mAP, and so on) by running
```python eval.py```

Remember that you don't need to adjust the gradient or other things when testing, just do as usual.
&nbsp;

## Demo explanation
<div  align="center">    
<img src="demo/demo_guitar.PNG" width = "80%" />
</div>
<div  align="center">    
<img src="demo/demo_snow.PNG" width = "80%" />
</div>
As shown in above picture, 'playing guitar' is a class that audio surpasses visual modality for most samples ('shovelling show' is just opposite), and we can tell audio achieves more adequate training and leads the optimization process. Our OGM-GE (as well as OGM) gains improvement in both modalties as well as multimodal performance, and the weak visual gains more porfit. The evaluation metric used in 'audio' and 'visual' is the predicted accuracy with classification scores just from one specific modality. 



## Citations
If you find this work useful, please consider citing it.

<pre><code>
@ARTICLE{Peng2022Balanced,
  title	= {Balanced Multimodal Learning via On-the-fly Gradient Modulation},
  author	= {Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, Di Hu},
  journal	= {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year	= {2022},
  Url	= {}
}
</code></pre>

## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.


## Contact us

If you have any detailed questions or suggestions, you can email us:
**xiaokangpeng@ruc.edu.cn**
