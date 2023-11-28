# Dual Graph Networks
[Dual Graph Networks for Pose Estimation in Crowded Scenes](https://rdcu.be/dsaP3)(IJCV2023)  

# Dependencies  
- PyTorch(>=1.0 && <=1.13.1)  
- mmcv(1.x)
- OpenCV
- visdom 
- pycocotools

This code is tested under Ubuntu 16.04, CUDA 10.1, cuDNN 7.1 environment with four NVIDIA TitanXP GPUs.

Python 3.6.5 version and PyTorch 1.3.1 is used for development.

# Quick Start under CrowdPose Datasets

All settings about CrowdPose Datasets please refer to [OPEC-Net project](https://github.com/lingtengqiu/OPEC-Net)

## Our trained weights and results

Our results can be downloaded in [Baidu](https://pan.baidu.com/s/1H-z90dd19ASaY7Thma92bQ )  
Extraction code: `mtfc`

The `results.zip` has two files, one is the weights we trained named `best_checkpoint.pth`, the other is keypoint result named `crowdpose_keypoints_results.json`.

**verify the results**  
make an directory named `CrowdPose_ADGN_withFlip_60epochs_bothConfidenceScoreIn_normalizedHeatmaps_batch32` under the directory `checkpoints`. Then, copy the weight file `best_checkpoint.pth` into the directory. Then one can run test script (see below) to verify our results.

**training log**  
our training log can be seen in `./log`, it is in tensorboard format. If one wants to see this result, one should install tensorboard first, then run the following command:
```
tensorboard --logdir CrowdPose_ADGN_withFlip_60epochs_bothConfidenceScoreIn_normalizedHeatmaps_batch32 --bind_all
```

 
## Training script
 e.g.  
 ```
 TRAIN_BATCH_SIZE=14
 CONFIG_FILES=./configs/OPEC_GCN_GrowdPose_Test_FC.py
 bash train.sh ${TRAIN_BATCH_SIZE} ${CONFIG_FILES} 
 ```
 after training, the result of CrowdPose is save into checkpoints/name/mAP.txt  
 the format of results like:
 ```
 epoch (without best match) (use best match) 
 ```
 We also offer logs in form of tensorboard chart in log/name, it can be invoked by the command
 ```
 tensorboard --logdir log/name/ --bind_all
 ```

## Test script
e.g.  
```
CHECKPOINTS_DIRS='path to your checkpoints files'
CONFIG_FILES =./configs/OPEC_GCN_GrowdPose_Test_FC.py
bash test.sh ${CHECKPOINTS_DIRS} ${CONFIG_FILES}
```

# Results

## Result on CrowdPose-test:  

Method | mAP@50:95 | AP50 | AP75 
:--:|:--:|:--:|:--:
Mask RCNN | 57.2 | 83.5 | 60.3 
Simple Pose | 60.8 | 81.4 | 65.7 
CrowdPose | 66.0 | 84.2 | 71.5 
AlphaPose+ | 68.5 | 86.7 |73.2 
HRNeXt | 70.4 | 84.1 | 75.6 
OPEC-Net | 70.6| 86.8 | 75.6 
**Ours** (joint branch) | **71.6** | - 
**Ours** (after fusion of both branches) |  **72.4** | 85.6 | **76.6**

**Note:**  
　　Ours (joint branch) shows the result being outputted from the joint branch of DGN.  
　　Ours (after fusion of both breanches) shows the result after fusion post-process. The fusion process is detailed on the paper. Any one can code it out from the steps we offer on the paper. We will update the post-process code in the near future.


## Result on COCO 2017 test-dev

Method | mAP@50:95 | AP50 | AP75 
:--:|:--:|:--:|:--:
Mask RCNN | 63.1 | 87.3 | 68.7
HigherHRNet | 70.5 | 89.3 | 77.2
CPN | 72.1 | 91.4 | 80.0
AlphaPose+ | 72.2 | 90.1 | 79.3
RMPE | 72.3 | 89.2 | 79.1
PoseFix | 73.6 | 90.8 | 81.0
Simple Baselines | 73.7 | 91.9 | 81.8
OPEC-Net (Simple Baselines) | 73.9 | 91.9 | 82.2
HRNet | 75.5 | **92.5** | 83.3
Ours (Simple Baselines) | 74.0 | 91.7 | 82.2
Ours (HRnet) | **75.7** | 92.3 | 83.3

# Citation
If you find our works useful in your reasearch, please consider citing:  
```
@article{tu2023dual,
  title={Dual Graph Networks for Pose Estimation in Crowded Scenes},
  author={Jun Tu, Gangshan Wu & Limin Wang},
  journal={International Journal of Computer Vision},
  year={2023}
}
```

