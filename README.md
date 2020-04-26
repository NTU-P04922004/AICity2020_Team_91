# 2020 AI City Challenge Track 4 - Team_91

## Dependencies

* ```Ubuntu 18.04```
* ```ffmpeg v4.1.4```
* ```Python v3.6.8```
* ```Opencv-python v4.1```
* ```PyTorch v1.3.1```
* ```Detectron2 v0.1.1```

## Training car detector

Facebook's [Detectron2](https://github.com/facebookresearch/detectron2) is used to train a car detector. The training data is the official challenge track #4 training data. Since there is no car bounding box labels for the training data, a pretrained car detector from the winner of the 2019 AI City Challenge (refer to reference #1 below) is used to predict pseudo car bounding box labels. Finally, the pseudo labels are used as ground truths to train the car detector.

The pre-trained models can be found here:
https://drive.google.com/open?id=1lw7ozQ6khVp01EQBytCdDTEXewUBAJaQ

The car bounding box labels for all test videos can be found here:
https://drive.google.com/open?id=13JHYxBHmNHXQQn3-cNzBbJ8kQr-0rQuF

### Generating ground truth for training a car detector

Generating ground truths for training a car detector

## Pipeline

1. Video frames extraction
2. Background and foreground analysis
3. ROI analysis
4. Car detection
5. Anomaly detection

## Background and foreground analysis

In this stage, static backgrounds and dynamic foregrounds for each frame in a video will be extracted. Static backgrounds are objects that remain static for a long duration in the video while foregrounds are objects constantly moving in the video.

OpenCV's Gaussian Mixture-based method (BackgroundSubtractorMOG2) is used to model backgrounds and foregrounds in a video.

## ROI analysis

In this stage, ROIs in a video will be generated as a binary mask. The ROIs are road regions that cars moving on it, therefore, extracted foreground information is used to model ROIs.

## Car detection

In this stage, a pretrained car detector is used to detect all cars in the background images of a video (generated in stage 2). The ROI mask generated from stage 3 will be used to filter false positives.

## Anomaly detection

In this stage, we want to analyze the car detection results from stage 4 and find out stalled cars. A backtracking algorithm is used to figure out the start and end times of anomaly events in a video.

## Reproduce challenge result

* Setup workspace
  * Create a workspace directory, for example, "```workspace```"
  * Put all test video files under the workspace directory
* Modify variables in ```bash.sh```
  * ```WORKSPACE_PATH```: path to the workspace directory
  * ```START_IDX```: index of the first video file (assumed that video files start from 1.mp4 to 100.mp4)
  * ```END_IDX```: index of the last video file
  * ```PRETRAINED_MODEL_PATH```: path to the pretrained car detection model
* Run ```bash.sh```
* After running ```bash.sh```, the challenge result will be saved as ```result.txt```

## References

1. https://github.com/ShuaiBai623/AI-City-Anomaly-Detection
2. https://github.com/jiayi-wei/NVIDIA-AICITY19-track3-anomaly-detection/blob/master/preprocess/make_mask.py
