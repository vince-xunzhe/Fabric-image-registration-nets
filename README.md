# Image-Registering-Nets
Unsupervised, deformable or non-rigid image image registration. (ResNets backbone) 

Keywords: `Deformable distorntion`, `Fabric Dewarping`, `Deep Learning`, `Computer Vision`. 


## 1. Introduction: 

This repository is about using DL for fabric image registration or alignmnet.

CNN (Convolutional Resnet) is trained to generate a robust bilinear resampler, which could restore the intrinsic warped texture.
Pros: (1) unsupervised learning strategies, so no need to labelling; (2) no need to iterative optimized during inference. Thus is time-saver for both development and futher deployment. (3) made Resnet convoluational can be a help for limitations of input image size, so no need to bring in more complicated structures wrt size and channels.    

## 2. Loss function

Instead of use the traditional loss, I intergrated similarity metric and mse loss into loss function, which we called joint similarity loss. 

## 3. Use case

Paper and text dewarp;
Fabric dewarp such as cloth, scarf;
Medical imaging registration such as MRI or CT;
...

