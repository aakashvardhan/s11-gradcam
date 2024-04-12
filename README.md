# Interpretability of ResNet18 on CIFAR10 using Grad-CAM

## Table of Contents
1. [Introduction](#introduction)
2. [Grad-CAM](#grad-cam)
3. [Interpretability of ResNet18 on CIFAR10](#interpretability-of-resnet18-on-cifar10)
4. [Results](#results)
5. [Conclusion](#conclusion)

## Introduction
ResNet18 is a convolutional neural network that is 18 layers deep. It has been trained on the CIFAR10 dataset which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. Interpretability of ResNet18 is difficult due to its depth and complexity. In this project, we use Grad-CAM to visualize the regions of the image that are important for classification.

## Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that uses the gradients of the target concept (say 'car' or 'plane' from CIFAR-10) with respect to the convolutional feature maps of the network. It produces a coarse localization map highlighting the important regions in the image for predicting the concept. This is helpful in understanding the model's decision-making process through deeper layers of the network.

## Interpretability of ResNet18 on CIFAR10
We use the ResNet18 model on CIFAR10 and Grad-CAM to visualize the regions of the image that are important for classification. A custom `pytorch-grad-cam` library is used to generate the Grad-CAM visualizations. This is taken from the [GitHub repository](https://github.com/jacobgil/pytorch-grad-cam) by Jacob Gildenblat. The Grad-CAM class functions from this repository is utilized to generate the visualizations.

### ResNet18 Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
```

### Grad-CAM output from Misclassified Images

**Misclassified Image 1: model.layer2[-2]**

![Misclassified Image 1](https://github.com/aakashvardhan/s11-gradcam/blob/main/asset/model_layer2_gradcam.png)

**Misclassified Image 2: model.layer3[-2]**

![Misclassified Image 2](https://github.com/aakashvardhan/s11-gradcam/blob/main/asset/model_layer3_gradcam.png)

**Misclassified Image 3: model.layer4[-1]**

![Misclassified Image 3](https://github.com/aakashvardhan/s11-gradcam/blob/main/asset/model_layer4_gradcam.png)

#### Observations

1. The Grad-CAM visualizations show the regions of the image that are important for classification.
2. The earlier layers focus on the edges and textures of the object or background.
3. The deeper layers focus on the more abstract features of the object, which indicates that the ResNet18 model is learning high-level features.

## Results

```
EPOCH: 1
Loss=1.3299800157546997 Batch_id=390 Accuracy=39.88: 100%|██████████| 391/391 [00:49<00:00,  7.83it/s]
Max Learning Rate: 0.0004370127942681678
Test set: Average loss: 1.8692, Accuracy: 4434/10000 (44.34%)

EPOCH: 2
Loss=1.0045512914657593 Batch_id=390 Accuracy=57.25: 100%|██████████| 391/391 [00:49<00:00,  7.83it/s]
Max Learning Rate: 0.0008530255885363355
Test set: Average loss: 1.0962, Accuracy: 6259/10000 (62.59%)

EPOCH: 3
Loss=0.8947468996047974 Batch_id=390 Accuracy=65.61: 100%|██████████| 391/391 [00:49<00:00,  7.86it/s]
Max Learning Rate: 0.0012690383828045033
Test set: Average loss: 1.0499, Accuracy: 6549/10000 (65.49%)

EPOCH: 4
Loss=0.8309239149093628 Batch_id=390 Accuracy=70.03: 100%|██████████| 391/391 [00:49<00:00,  7.85it/s]
Max Learning Rate: 0.001685051177072671
Test set: Average loss: 0.7918, Accuracy: 7404/10000 (74.04%)

EPOCH: 5
Loss=0.6489930152893066 Batch_id=390 Accuracy=72.72: 100%|██████████| 391/391 [00:49<00:00,  7.85it/s]
Max Learning Rate: 0.002099641979539642
Test set: Average loss: 0.7913, Accuracy: 7405/10000 (74.05%)

...
...
...

EPOCH: 15
Loss=0.4032979905605316 Batch_id=390 Accuracy=87.48: 100%|██████████| 391/391 [00:49<00:00,  7.89it/s]
Max Learning Rate: 0.0006997819795396419
Test set: Average loss: 0.3732, Accuracy: 8821/10000 (88.21%)

EPOCH: 16
Loss=0.31994548439979553 Batch_id=390 Accuracy=88.90: 100%|██████████| 391/391 [00:49<00:00,  7.88it/s]
Max Learning Rate: 0.000559795979539642
Test set: Average loss: 0.3492, Accuracy: 8898/10000 (88.98%)

EPOCH: 17
Loss=0.2587328553199768 Batch_id=390 Accuracy=90.19: 100%|██████████| 391/391 [00:49<00:00,  7.89it/s]
Max Learning Rate: 0.00041980997953964204
Test set: Average loss: 0.3357, Accuracy: 8972/10000 (89.72%)

EPOCH: 18
Loss=0.2883536219596863 Batch_id=390 Accuracy=91.47: 100%|██████████| 391/391 [00:49<00:00,  7.85it/s]
Max Learning Rate: 0.0002798239795396421
Test set: Average loss: 0.2794, Accuracy: 9146/10000 (91.46%)

EPOCH: 19
Loss=0.17770078778266907 Batch_id=390 Accuracy=93.06: 100%|██████████| 391/391 [00:49<00:00,  7.88it/s]
Max Learning Rate: 0.00013983797953964197
Test set: Average loss: 0.2744, Accuracy: 9182/10000 (91.82%)

EPOCH: 20
Loss=0.3206477165222168 Batch_id=390 Accuracy=94.16: 100%|██████████| 391/391 [00:49<00:00,  7.87it/s]
Max Learning Rate: -1.4802046035796226e-07
Test set: Average loss: 0.2640, Accuracy: 9214/10000 (92.14%)
```

![Plot](
