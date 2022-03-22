# Conv2d layer & Dense layer & Softmax layer

## Intro

This note use the dataset MNIST from tensorflow tutorial's example. Using conv2d layers, dense layers, max pooling layers and softmax layer to build the network to predict the class of the flower and The loss function using here is cross entropy. The accuracy can be up to 98% (varied)

## Main Reference

- [TensorFlow中文社區](https://doc.codingdict.com/tensorflow/tfdoc/tutorials/mnist_beginners.html)

## Required

* language
  * Python
* Package
  * TensorFlow (**1.14.0** or higher)
  * Numpy

## Content

- MNIST Dataset
- Single Softmax Layer
- Conv2d + Max Pooling + Dense + Softmax
- The loss function (Cross Entropy)

### Import packages

Import TensorFlow

```python

import tensorflow as tf

```

### MNIST Dataset

Download the MNIST under your current directory, then read the dataset and store as **mnist**

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

The MNIST dataset contain thousands of hand writing pictures and their classes.  
<img src="https://user-images.githubusercontent.com/92711171/159486789-8de2a95e-57c9-4cc4-8016-f003e84f5a61.png" width="30%"/>














