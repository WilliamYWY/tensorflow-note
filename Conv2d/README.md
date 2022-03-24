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

## Single Softmax Layer

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

Let's at first explore the training data.  
Use the following code to get the train data as the batch size you want.
```python
batch = mnist.train.next_batch(1) # get only one set of data
images = batch[0] #array of images
labels = batch[1] #array of labels

```
Each images have been resized to [784] with 0 and 1.  [0,0,0,1,1,.....] (We will resize this array to 28*28 for conv2d input later).  
And the size of labels is [10] with 0 and 1. [0,0,0,0,0,1,0,0,0,0] (one-hot encoding)

### softmax layer

Let's first try using one softmax layer to predict the type of input pictures.  
In brief, softmax function will output the probability of each class, which sum to 1.0. [0.1, 0.05, 0.25, .....]  
(More information checkout https://en.wikipedia.org/wiki/Softmax_function)  
  
First we create a **placeholder** to declare the size of input we want which let our graph to build.  
**None** is the **batch size** which we prefer to declare as unknown so we can modified later.
```python
x = tf.placeholder("float", [None, 784]) 
```
  
For the softmax function, we have to provide the equation.  
**W** is the weight and **b** is the bias.  
  
- **tf.Variable** help to create and initiate  **Tensor**  
- **tf.zeros** will create an tensor full of 0 of the input shape.    
  
While we directly input data to the softmax layer, the first position of the shape of the weight should be **the size of a image**,  
and since the classes that we predict are 10 (means the output) the shape of the weight should be [784, 10].  
The shape of bias will follow the numbers of output (out channel) so it would be [10].  
  
- **tf.matmul** help multiply two tensor W and x (input).  

The equation would be  __W * x + b__.  
Then we input to the softmax function.  
y is our output answer(predict).

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)
```
  
For classification problem we often use **Cross Entropy** as loss funcition to train the model.  
y_ is a placeholder which will be filled with **labels**(correct answer).  
We will discuss the cross entropy function more in the end of this note.  
  
We will also need an **optimizer** to regulate our training, which in here we use the **GradientDescentOptimizer** and the learning rate being set as 0.01.  
The goal of the model will be **"try as hard as it can to reduce the cross entropy"**, so we use minimize function.
```python
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #loss function
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #choose optimizer and load loss function
```
  
Finially we can start to feed our data to the single softmax layer network that we just created.  
To start the training, we have to create a **session** first.  
Then we initialize all the variables. (Remember to init first every time you start a new training)
```python
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #--- more code ----#
```
  
We can decide how many epochs that we want our network to be train by using for loop.  
In the second line of the code we generate 100 training data for each epoch.
Then we load our operation function (train_step) and our data to the session.  
  
Remember that we have created two placeholder x and y_, here we finially filled them!
```python
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 500
    for i in range(epoch): #epoch
        batch = mnist.train.next_batch(100) #generate 100 data for one epoch
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[0]}) #feed 
```
Since the output of optimizer is None, you won't see anything during the training even if you try to print out.  
  
Lastly we'll evaluate our model with its accuracy.  
Alike the loss function cross entropy we create before, over here we create a function to test its accuracy.  
  
- **tf.argmax** will out the index of the maximum value in the array. (tf.argmax([1,2,3,4,5] will output 4)
- **tf.equal** will out put an array which store the True or False value.  
If the two input array have the same value at same index, the value will be True. (tf.equal([1,2,3],[2,2,2]) will output [False, True, False])  
- **tf.cast** help to transfer the True and False to 1 and 0.  
- **tf.reduce_mean** will ouput the mean of the array.
Over here we don't apply optimizer since we are not going to train.
```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```
  
Again we create a session and run the operation and feed in the **test** data.
Since we apply the accuracy operation (without optimizer) this do have an output which is the accuracy that we want to know.
```python
witth tf.Session() as sess:
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```



















