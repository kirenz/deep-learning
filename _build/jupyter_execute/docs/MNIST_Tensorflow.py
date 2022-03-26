#!/usr/bin/env python
# coding: utf-8

# # MNIST TensorFlow
# 
# *The following code example is mainly based on Mikhail Klassen's article [Tensorflow vs. PyTorch by example](https://towardsdatascience.com/tensorflow-vs-pytorch-by-example-66d37901c663)*
# 
# ## MNIST
# 
# The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large database of handwritten digits that is commonly used for training various image processing systems.
# 
# ![](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)
# 
# Image source: [Wikipedia, Josef Steppan](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

# ## Setup

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


# In[7]:


# Random seed for reproducibility
seed = 42

tf.keras.utils.set_random_seed(seed)

# Save the model at the end?
save_model = False

# Batch sizes for training and testing
batch_size = 64
test_batch_size = 14

# Training epochs (usually 10 is a good value)
n_epochs = 2

# Learning rate
learning_rate = 1.0

# Decay rate for adjusting the learning rate
gamma = 0.7

# Number of target classes in the MNIST data
num_classes = 10

# Data input shape
input_shape = (28, 28, 1)


# In[8]:


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The scaled mean and standard deviation of the MNIST dataset (precalculated)
data_mean = 0.1307
data_std = 0.3081

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], 
                          x_train.shape[1], 
                          x_train.shape[2], 1)

x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2], 1)

# Normalize the data
x_train = (x_train/255.0 - data_mean) / data_std
x_test = (x_test/255.0 - data_mean) / data_std

# Convert labels to one-hot vectors
y_train = tf.one_hot(y_train.astype(np.int32), depth=num_classes)
y_test = tf.one_hot(y_test.astype(np.int32), depth=num_classes)


# In[9]:


# Define the architecture of the neural network
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                      padding='valid', 
                                      activation='relu',
                                      input_shape=input_shape),
    tf.keras.layers.Conv2D(64, (3,3), strides=(1,1),
                                      padding='valid',
                                      activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# In[10]:


# Decay the learning rate at a base rate of gamma roughly every epoch, which
# is len(x_train) steps
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=len(x_train),
    decay_rate=gamma)

# Define the optimizer to user for gradient descent
optimizer = tf.keras.optimizers.Adadelta(scheduler)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

# Display a model summary
model.summary()


# In[11]:


# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_data=(x_test, y_test),
          validation_batch_size=test_batch_size)


# In[12]:


if save_model:
    model.save_weights("mnist_cnn_tf.ckpt")

