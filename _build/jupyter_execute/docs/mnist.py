#!/usr/bin/env python
# coding: utf-8

# # MNIST data

# *The content of this notebook is based on the official TensorFlow tutorial "Basic classification: Classify images of clothing". 

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## Import data

# We use the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset which contains 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28 by 28 pixels), as seen here:
# 
# <table>
#   <tr><td>
#     <img src="https://tensorflow.org/images/fashion-mnist-sprite.png"
#          alt="Fashion MNIST sprite"  width="600">
#   </td></tr>
#   <tr><td align="center">
#     <b>Figure 1.</b> <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST samples</a> (by Zalando, MIT License).<br/>&nbsp;
#   </td></tr>
# </table>
# 
# Fashion MNIST is intended as a drop-in replacement for the classic [MNIST](http://yann.lecun.com/exdb/mnist/) dataset—often used as the "Hello, World" of machine learning programs for computer vision. The MNIST dataset contains images of handwritten digits (0, 1, 2, etc.) in a format identical to that of the articles of clothing you'll use here.
# 
# Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images. You can access the Fashion MNIST directly from TensorFlow. Import and [load the Fashion MNIST data](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data) directly from TensorFlow:

# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Loading the dataset returns four NumPy arrays:
# 
# * The `train_images` and `train_labels` arrays are the *training set*—the data the model uses to learn.
# * The model is tested against the *test set*, the `test_images`, and `test_labels` arrays.
# 
# The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. The *labels* are an array of integers, ranging from 0 to 9. These correspond to the *class* of clothing the image represents:
# 
# <table>
#   <tr>
#     <th>Label</th>
#     <th>Class</th>
#   </tr>
#   <tr>
#     <td>0</td>
#     <td>T-shirt/top</td>
#   </tr>
#   <tr>
#     <td>1</td>
#     <td>Trouser</td>
#   </tr>
#     <tr>
#     <td>2</td>
#     <td>Pullover</td>
#   </tr>
#     <tr>
#     <td>3</td>
#     <td>Dress</td>
#   </tr>
#     <tr>
#     <td>4</td>
#     <td>Coat</td>
#   </tr>
#     <tr>
#     <td>5</td>
#     <td>Sandal</td>
#   </tr>
#     <tr>
#     <td>6</td>
#     <td>Shirt</td>
#   </tr>
#     <tr>
#     <td>7</td>
#     <td>Sneaker</td>
#   </tr>
#     <tr>
#     <td>8</td>
#     <td>Bag</td>
#   </tr>
#     <tr>
#     <td>9</td>
#     <td>Ankle boot</td>
#   </tr>
# </table>
# 
# Each image is mapped to a single label. Since the *class names* are not included with the dataset, store them here to use later when plotting the images:

# In[3]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ## Explore the data
# 
# Let's explore the format of the dataset before training the model. The following shows there are 60,000 images in the training set, with each image represented as 28 x 28 pixels:

# In[4]:


train_images.shape


# Likewise, there are 60,000 labels in the training set:

# In[5]:


len(train_labels)


# Each label is an integer between 0 and 9:

# In[6]:


train_labels


# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:

# In[7]:


test_images.shape


# And the test set contains 10,000 images labels:

# In[8]:


len(test_labels)


# ## Preprocess the data
# 
# The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

# In[9]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255. It's important that the *training set* and the *testing set* be preprocessed in the same way:

# In[10]:


train_images = train_images / 255.0

test_images = test_images / 255.0


# To verify that the data is in the correct format and that you're ready to build and train the network, let's display the first 25 images from the *training set* and display the class name below each image.

# In[11]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

