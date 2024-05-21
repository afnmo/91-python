# 91
This repository contains the models `extract_plate.pt` which you can find [here](https://drive.google.com/file/d/1AnQbLbE4w9mcUoCK_TBRthUFy6m1oN8p/view?usp=drive_link) and `locate_plate.pt` which you can find [here](https://drive.google.com/file/d/1IQyfaDGW6BhoCUtQdd1-eUgqPOL_hM0g/view?usp=drive_link). 
Ensure that both files are added to the / path.

The source code of 'extract_plate.pt' can be found [here](https://colab.research.google.com/drive/1SysWSzIrbtFweXBg3VXk5tI6tkFWz5uO?usp=drive_link).
The source code of 'locate_plate.pt' can be found [here](https://colab.research.google.com/drive/1wvpJgqDZNke_PFvFztu-Z6ffjgZuqEx8?usp=drive_link).

<p align="center">
<a href="https://www.youtube.com/watch?v=fyJB1t0o0ms">
    <img width="600" src="https://utils-computervisiondeveloper.s3.amazonaws.com/thumbnails/with_play_button/anpr_yolo2.jpg" alt="Watch the video">
    </br>Watch on YouTube: Automatic number plate recognition with Python, Yolov8, and EasyOCR!
</a>
</p>

## data

The video I used in this tutorial can be downloaded [here](https://drive.google.com/file/d/12sBfgLICdQEnDSOkVFZiJuUE6d3BeanT/view?usp=sharing).

## models

A Yolov8 pre-trained model was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and following this [step by step tutorial on how to train an object detector with Yolov8 on your custom data](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide). 

The trained model is available on my [Patreon](https://www.patreon.com/ComputerVisionEngineer).

## dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort) as mentioned in the [video](https://youtu.be/fyJB1t0o0ms?t=1120).
