#  Recycling and Waste Image Classification
This project implements a waste image classification system using the NVIDIA Jetson Inference Library. The model has been retrained on a custom dataset to classify items into two main categories — Recycling and Waste — with high accuracy.
## Overview
The system can classify waste into two distinct categories, making it suitable for real-world applications in waste management, recycling facilities, educational campaigns, and even smart bin systems. By providing quick and accurate classification, the model helps reduce sorting mistakes and supports more effective recycling practices.
## Why I Chose This Project
Living in Canada, I’ve noticed just how much garbage we throw away — and the numbers really drive it home. According to the ([Government of Canada](https://www.canada.ca/en/environment-climate-change/services/environmental-indicators/solid-waste-diversion-disposal.html)) in 2022, Canadians produced **36.5 million tonnes** of solid waste, and over **72%** of it ended up in landfills or incinerators . That works out to about **684 kilograms per person** in just one year, which is one of the highest rates in the world.
The thing is, even with all our recycling programs, a lot of recyclable stuff still ends up in the trash. Most of the time, it’s not because people don’t care — it’s because they’re not sure where something actually belongs. That uncertainty is what made me want to create this AI waste classification model. The idea was to make a simple tool that could look at a picture of an item and tell you whether it should go in the recycling bin or the garbage. It’s nothing overly complicated, but I think small tools like this could make a big difference in helping people sort waste correctly.

<img width="1280" height="554" alt="1728415349801" src="https://github.com/user-attachments/assets/b510320e-82e9-4d6e-920d-01bc8a087d82" />

### The Dataset

[The model](https://drive.google.com/file/d/1mUO3UONC8jDlJMobr92M0ySdrRZzDbyC/view?usp=sharing) uses a ResNet-18 architecture that has been retrained on [this refined waste classification dataset](https://drive.google.com/file/d/1JVh1xh__7XxJc5QG-dvBXSFSzIcl0spY/view?usp=sharing). For training, 2,456 images were used for both categories (recycling and waste-products), 290 were used for validation, and 145 were used for testing.



I kept my dataset simple. Instead of a lot of specific categories, I split everything into just two groups:

Recycling — plastics, cardboard, paper, and other materials most cities accept in their recycling programs.

Waste — food scraps, dirty packaging, and things that can’t be recycled locally.


In total, I had about 5,800 images, evenly divided between these two categories. I labeled each image manually so the model had clear, accurate examples to learn from. While this approach meant the AI couldn’t give very specific labels like “glass bottle” or “plastic fork,” it allowed it to focus on the main question: Is this recyclable or not?
## How the Model Works
I built the model using transfer learning, which means I started with a pre-trained neural network and fine-tuned it on my dataset. I chose ResNet-18 because it’s known for being accurate and efficient. I used data augmentation — things like rotating, flipping, and changing lighting — so the model would be better at handling real-world photos.
The model outputs a simple binary classification: “Recycling” or “Waste,” along with a confidence score. Keeping the categories broad actually helped prevent it from overfitting to tiny, irrelevant details and kept it focused on making the right overall decision.
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
jetson-inference/python/training/classification/data/recycing_waste-products/
├── train/
│   ├── waste-products/
│   ├── recycling/
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
  python3 train.py --model-dir=models/recycing_waste-products data/Data
  ```
3. Export Model
  ```
  # Still in docker container:
  python3 onnx_export.py --model-dir=models/recycing_waste-products
  ```

## Using the Model

### Set Variables
```
cd jetson-inference/python/training/classification
NET=models/recycing_waste-products
DATASET=data/Data
```

### Test on Image
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/boot/<image.jpg> result.jpg
```
Replace <image.jpg> with your actual image.

![output](https://github.com/user-attachments/assets/0c817494-a5f7-4d17-89f5-32ad0dd7291d)

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
