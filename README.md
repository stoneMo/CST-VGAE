
# Long-term Head Pose Forecasting Conditioned on the Gaze-guiding Prior

**[accepted to CVPR Workshop 2021]**

![alt text](https://github.com/stoneMo/CST-VGAE/blob/imgs/title_image.png?raw=true)

## Abstract

Forecasting head pose future states is a novel task in computer vision. Since future may have many possibilities, and the logical results are much more important than the impractical ones, the forecasting results for most of the sce- narios should be not only diverse but also logically realistic. These requirements pose a real challenge to the current methods, which motivates us to seek for better head pose representation and methods to restrict the forecasting reasonably. In this paper, we adopt a spatial-temporal graph to model the interdependencies between the distribution of landmarks and head pose angles. Furthermore, we propose the conditional spatial-temporal variational graph autoencoder (CST-VGAE), a deep conditional generative model for learning restricted one-to-many mappings conditioned on the spatial-temporal graph input. Specifically, we improve the proposed CST-VGAE for the long-term head pose forecasting task in terms of several aspects. First, we introduce a gaze-guiding prior based on the physiology. Then we apply a temporal self-attention and self-supervised learning mechanism to learn the long-range dependencies on the gaze prior. To better model head poses structurally, we introduce a Gaussian Mixture Model (GMM), instead of a fixed Gaussian in the encoded latent space. Experiments demonstrate the effectiveness of the proposed method for the long-term head pose forecasting task. We achieve superior forecasting performance on the benchmark datasets compared to the existing methods.

## Requirements


To install requirements, you can:

```
pip install -r requirements.txt

```

Note that our implementation is based on Python 3.7, and PyTorch deep learning framework, trained on NIVIDA GeForce RTX 2080 Ti in Ubuntu 16.04 system.

## Codes

There are three different section of this project. 
1. Data pre-processing
2. Training and testing 
3. Pretrained models

We will go through the details in the following sections.

### 1. Data pre-processing

In this work, we demonstrate the effective of the proposed CST-VGAE for long-term forecasting problems on BIWI Kinect Head Pose Database. The BIWI dataset contains 24 videos of 20 subjects in the controlled environment. There are a total of roughly 15, 000 frames in the dataset.

If you don't want to re-download every dataset images and do the pre-processing again, or maybe you don't even care about the data structure in the folder. Just download the file **data.zip** from the following link, and replace the data subfolder in the data_preprocessing folder.

[Google drive](https://drive.google.com/drive/folders/1T8mhPQcVhbudZg2LwCvgxnwyUJO_dS4y?usp=sharing)

Now you can skip to the "Training and testing" stage. If you want to do the data pre-processing from the beginning, you need to download the dataset first, and unzip the dataset in the dataset folder for data_preprocessing.

#### Download the datasets

+ [BIWI Kinect Head Pose Database](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

#### [Details]

Before we go through the details on generating the training and validation data, we need to extract all landmarks, gazes(pupils), and head poses from the BIWI dataset. We implement [Fan](https://github.com/1adrianb/face-alignment), one state-of-art face alignment tool to extract 19 facial landmarks from each frame, and utilize [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), one state-of-art landmarks detector to extract pupil keypoints. 

Notice that, some edge cases exist in certain frames, where just one pupil can be detected because of the angle of the head pose. In order to make the number of facial landmarks consistent with other frames, i.e., 21, we empirically set the x coordinate of the center landmark on the nose as the x coordinate of the undetected pupil. And the y coordinate of the detected pupil landmark represents the y coordinate of the unseen pupil landmark. 

Please remember to download the file **body_pose_model.pth** from the following link, and put it in the model subfolder in the data_preprocessing folder. 

[Google drive](https://drive.google.com/drive/folders/1fvsywqKLSi83V4tf7W7idx7lyWQR6ZRd?usp=sharing)

In the experiments, we implement the proposed CST-VGAE model on long-term (multi-frame) forecasting problems. Here, our goal is to generate long-term head poses (30 frames) given five past frames as input.

After download all these aforementioned files, you can run
```
bash gen_data.bash

```
NOTE: MAKE SURE TO IMPLEMENT THIS STEP ON GPUS, AS IT WILL SAVE YOUR TIME A LOT.


### 2. Training and testing 
```

# Training
# Make sure SKIP_TRAINING=0 in the shell script. 

bash full_train_test.bash


# Testing
# Make sure SKIP_TRAINING=1, --pretrain 1\ in the shell script. 

bash full_train_test.bash

```

Just remember to check which model type you want to use in the shell and you are good to go. Note that we calculate the MAE of yaw, pitch, roll independently, and average them into one single MAE for evaluation. 

### 3. Pretrained models


```

You also can use the pretrained models to finetune or evaluate the model.

# Training
# Make sure SKIP_TRAINING=0, --pretrain 1\ in the shell script. 

bash full_train_test.bash


# Testing
# Make sure SKIP_TRAINING=1, --pretrain 1\ in the shell script. 

bash full_train_test.bash

```