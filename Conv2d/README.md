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

## Conv2d + Max Pooling + Dense + Softmax

Now let's try a more complex model with two convolution layer, two max pooling layer, one dence layer and one softmax layer.  
  
First we will create several function that help create different layer more conveniently.
- ### Conv2d layer
We will not discus how convolution layer work over here!  
Instead we'll talk about what important element the convolution network need.  
For a conv2d layer, we will need a kernel, a bias and also a activation function.

```python
def conv2d(x, output_channel, kernel, strides):
    """
    input kernel size [length, width]
    strides int
    """
    #kernal size for 2d [length, width, channel, features]
    kernel_shape = [kernel[0], kernel[1], x.get_shape()[-1].value, output_channel]
    initial_W = tf.truncated_normal(kernel_shape, stddev=0.1) #std some what IMPORTANT
    W = tf.Variable(initial_W)
    
    #Bias size = output_channel
    initial_B = tf.constant(0.1, shape=[output_channel])
    B = tf.Variable(initial_B)
    #build layer + Bias
    layer = tf.nn.conv2d(x, W, strides=[strides,strides,strides,strides], padding="SAME")

    #use activation func
    conv_act_layer = tf.nn.relu(layer + B)

    return conv_act_layer
```
  
Lets start with kernel.  
The kernel for a 2d convolution should have four elements:
- length of the kernel
- width of the kernel
- numbers of channel of input
- numbers of channel of desired output (filters)
  
Which bulid up as array [length, width, channel, output_channel].  
For the function input we only ask for the length and width for the kernel which saved in the para kernel.  
The output channel will be saved separately.  
>Note: x.get_shape()[-1].value will output the channel of input.  

We need to generate weight with the shape of kernel for our layer, in here we use **tf.truncated_normal**.  
>Note: tf.truncated_normal will generate random value from nomal distribution which within two standard error(stddev) from mean (in here 0).  

After we initialize the value, we will asign it to our variable scope to be utilized by our model.
>Note: tf.Variable will create a variable.

```python
#kernal size for 2d [length, width, channel, features]
kernel_shape = [kernel[0], kernel[1], x.get_shape()[-1].value, output_channel]
initial_W = tf.truncated_normal(kernel_shape, stddev=0.1) #std some what IMPORTANT
W = tf.Variable(initial_W)
```

Alike weight, now we create bias here with the shape [output_channel] (same as the last dimension of the kernel).  
We use **tf.constant** to initialize the value to 0.1, then we also asign it to our variable scope.

```python
#Bias size = output_channel
initial_B = tf.constant(0.1, shape=[output_channel])
B = tf.Variable(initial_B)
```

Then we can input parameters to build a conv2d layer.  
We set padding here as SAME to ensure that the image stay the same size after processing.  
>Note: strides moving steps = [batch, length, width, channel]
```python
layer = tf.nn.conv2d(x, W, strides=[strides,strides,strides,strides], padding="SAME")
```

Apply activation function to our layer and also add on bias.
```python
#use activation func
conv_act_layer = tf.nn.relu(layer + B)
```

Done! We can return **conv_act_layer** finally!

- ### Max poolong layer
Max pooling will help to reduce the size of data while retaining the most important information(Maximum value).  
Normally the kernel size will be set to [1, length, width, 1] for 2d max pooling.  
And the strides usually being set as same as kernel.

```python
def max_pool(x, kernel, strides):
    """
    input kernel size [length, width]
    strides int
    """
    max_pool_layer = tf.nn.max_pool(x, 
                       ksize=[1, kernel[0], kernel[1], 1], 
                       strides= [1, strides, strides, 1], 
                       padding="SAME")

    return max_pool_layer
```
We don't have to create weight and bias for this layer so we can just easily input the parameters then return the layer!
  
- ### Dense Layer
Dense layer alikes softmax layer needed a weight and a bias and also need to provide how many output channels we want.
```python
def dense_layer(x, output_channel):
    # weight
    shape = [x.get_shape()[-1].value, output_channel]
    initial_W = tf.truncated_normal(shape, stddev=0.1) #std some what IMPORTANT
    W = tf.Variable(initial_W)
    #bias
    initial_B = tf.constant(0.1, shape=[output_channel])
    B = tf.Variable(initial_B)

    layer = tf.matmul(x, W) # (W * x)
    act_layer = tf.nn.relu(layer + B)

    return act_layer
```

The shape of our weight will be [input's size, output_channel].  
Then we initialize it and create as a variable.
```python
# weight
shape = [x.get_shape()[-1].value, output_channel]
initial_W = tf.truncated_normal(shape, stddev=0.1) #std some what IMPORTANT
W = tf.Variable(initial_W)
```
Same as bias.
```python
#bias
initial_B = tf.constant(0.1, shape=[output_channel])
B = tf.Variable(initial_B)
```
Mutiply input with the weight and add on bias, then apply activation function.
```python
layer = tf.matmul(x, W) # (W * x)
act_layer = tf.nn.relu(layer + B)
```
Done! Return that act_layer !

- ### Softmax Layer
We have just talked  about the softmax layer previously, now we just need to wrap them into a function.  
```python
def softmax_layer(x, output_channel):
    # weight
    shape = [x.get_shape()[-1].value, output_channel]
    initial_W = tf.truncated_normal(shape, stddev=0.1) #std some what IMPORTANT
    W = tf.Variable(initial_W)
    #bias
    initial_B = tf.constant(0.1, shape=[output_channel])
    B = tf.Variable(initial_B)

    layer = tf.matmul(x, W)
    act_layer = tf.nn.softmax(layer+B)

    return act_layer
```

- ### Build up the Network
After we create several function, we can now use those modules to build our network!  
>Note: Below is the graph of our network.
<img src="https://user-images.githubusercontent.com/92711171/159866495-157eecbd-5212-4aad-acf1-4cc70999b599.png" width="30%"/>   
The sequance of the network:  
  
1. Conv2d
2. Max pooling
3. Conv2d
4. Max pooling
5. Dense
6. Softmax

```python
#input placeholder
x = tf.placeholder(dtype="float", shape=[None, 784])
input_x = tf.reshape(x, shape=[-1, 28, 28, 1])

#model network
conv2d_1 = conv2d(input_x, 32, [5,5], 1)
max_pool1 = max_pool(conv2d_1, [2,2], 2)
conv2d_2 = conv2d(max_pool1, 64, [5,5], 1)
max_pool2 = max_pool(conv2d_2, [2,2], 2)
flatten = tf.reshape(max_pool2, [-1, max_pool2.get_shape()[1]*max_pool2.get_shape()[2]*max_pool2.get_shape()[3]])
dense_1 = dense_layer(flatten, 1024)
output = softmax_layer(dense_1, 10) #output = 10
```
First we also need to create a placeholder for input.
However, this time we will need to reshape the data into a 2D array to fit in to our first conv2d layer.  
-1 means that the reshape function will refer the size itself regarding the input.  
>Note: If batch size = 20, the shape will be [20, 28, 28, 1]
```python
x = tf.placeholder(dtype="float", shape=[None, 784])
input_x = tf.reshape(x, shape=[-1, 28, 28, 1])
```

Then we arrange our layers.  
```python
#model network
conv2d_1 = conv2d(input_x, 32, [5,5], 1)
max_pool1 = max_pool(conv2d_1, [2,2], 2)
conv2d_2 = conv2d(max_pool1, 64, [5,5], 1)
max_pool2 = max_pool(conv2d_2, [2,2], 2)
flatten = tf.reshape(max_pool2, [-1, max_pool2.get_shape()[1]*max_pool2.get_shape()[2]*max_pool2.get_shape()[3]])
dense_1 = dense_layer(flatten, 1024)
output = softmax_layer(dense_1, 10) #output = 10

```
Before sending input to the dense_1 layer, we reshape(flatten) our output from the conv2d_2 layer  
from four dimension to two dimension in order to fit the input size of dense layer.

Great! Now we can again create loss function and ready to feed our data to the network!

- ### Loss Function and Train
Same as what we have done at the section of single layer, we create a cross entropy loss function for training and a accuracy function for testing.  
```python
#loss function
y_ = tf.placeholder("float", shape=[None, 10]) #label
cross_entropy = -1 * tf.reduce_sum(y_*tf.log(output)) #reduce_sum sum all the element
#add optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#test evaluation function (accuracy)
acc_matrix = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
acc_eval = tf.reduce_mean(tf.cast(acc_matrix, "float"))
```

Finally! Let's train our model!  
```python
# Train
epoch = 5000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch = mnist.train.next_batch(50)
        #visualize ACC every 50 epoch
        if i % 50 == 0:
            train_acc = acc_eval.eval(feed_dict={x: batch[0], y_:batch[1]})
            print(f"Steps{i}: ACC: {train_acc}")
        #train network
        train_step.run(feed_dict={x: batch[0], y_:batch[1]})
    test_acc = acc_eval.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels})
    print(f"Network ACC: {test_acc}")
```

We evaluate the accuracy every 50 epochs.  
**Remember to run the accuracy operation before training for that epoch, or the network can cheat!!!** 
>Note: acc_eval.eval is same as using session to run the operation

```python
if i % 50 == 0:
    train_acc = acc_eval.eval(feed_dict={x: batch[0], y_:batch[1]})
    print(f"Steps{i}: ACC: {train_acc}")
```
Eventually we can receive the accuracy of the prediction of our network which is around 90~98%!  

## Conclusion 
After finishing this practice, you might be able to understand the basic of conv2d network and how to build up functions that help output a layer!  
Give yourelf a big hand!👏👏👏👏








