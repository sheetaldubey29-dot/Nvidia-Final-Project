#  Recycling and Waste Image Classification
This project implements a waste image classification system using the NVIDIA Jetson Inference Library. The model has been retrained on a custom dataset to classify items into two main categories — Recycling and Waste — with high accuracy.
## Overview
The system can classify waste into two distinct categories, making it suitable for real-world applications in waste management, recycling facilities, educational campaigns, and even smart bin systems. By providing quick and accurate classification, the model helps reduce sorting mistakes and supports more effective recycling practices.
## Why I Chose This Project
Living in Canada, I’ve noticed just how much garbage we throw away — and the numbers really drive it home. According to the ([Government of Canada](https://www.canada.ca/en/environment-climate-change/services/environmental-indicators/solid-waste-diversion-disposal.html)) in 2022, Canadians produced **36.5 million tonnes** of solid waste, and over **72%** of it ended up in landfills or incinerators . That works out to about **684 kilograms per person** in just one year, which is one of the highest rates in the world.
The thing is, even with all our recycling programs, a lot of recyclable stuff still ends up in the trash. Most of the time, it’s not because people don’t care — it’s because they’re not sure where something actually belongs. That uncertainty is what made me want to create this AI waste classification model. The idea was to make a simple tool that could look at a picture of an item and tell you whether it should go in the recycling bin or the garbage. It’s nothing overly complicated, but I think small tools like this could make a big difference in helping people sort waste correctly.

<img width="1280" height="554" alt="1728415349801" src="https://github.com/user-attachments/assets/b510320e-82e9-4d6e-920d-01bc8a087d82" />


## The Algorithm

### Model Architecture

[The model](Link goes here) uses a ResNet-18 architecture that has been retrained on [this footwear classification dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images). For training, 4000 images were used for each category, 500 were used for validation, and 500 were used for testing.

### Footwear Categories
![add image descrition here](https://news.harvard.edu/wp-content/uploads/2024/11/cheese-thumbnail-min.png)

The model classifies footwear into the following 3 categories:

1. Boot
2. Sandal
3. Shoe

## Setup

### 1. Install Jetson Inference

```
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install
```

### 2. Prepare Dataset

Organize images like this:
```
jetson-inference/python/training/classification/data/footwear/
├── train/
│   ├── boot/
│   ├── sandal/
│   ├── shoe/
├── val/
└── test/

```

### 3. Training

1. Enable more memory: `echo 1 | sudo tee /proc/sys/vm/overcommit_memory`
2. Train the model (I used batch size of 4)
  ```
  cd jetson-inference
  ./docker/run.sh
  cd python/training/classification
  python3 train.py --model-dir=models/footwear data/footwear
  ```
3. Export Model
  ```
  # Still in docker container:
  python3 onnx_export.py --model-dir=models/footwear
  ```

## Using the Model

### Set Variables
```
cd jetson-inference/python/training/classification
NET=models/footwear
DATASET=data/footwear
```

### Test on Image
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/boot/<image.jpg> result.jpg
```
Replace <image.jpg> with your actual image.
### Live Camera
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt /dev/video0 output.mp4
```

### Process Video
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt input.mp4 output.mp4
```

## Resources
* [The Finished Model](https://drive.google.com/file/d/1rxGChiVVU55-F3HUiedWlTLn6W6AyduL/view?usp=sharing)
* [Dataset](https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images)
* [ImageNet Documentation](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md)
* [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)
* [Video Demonstration](https://youtu.be/iGFWuKGNE6Y)
