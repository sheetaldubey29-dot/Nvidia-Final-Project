# Footwear Image Classification

This project implements a footwear image classification system using NVIDIA Jetson Inference Library. The model has been retrained on a comprehensive footwear dataset to classify different types of shoes with high accuracy.

## Overview

The system can classify footwear into 3 distinct categories, making it suitable for applications in e-commerce, inventory management, fashion analysis, and automated retail systems.

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
