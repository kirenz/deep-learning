#!/usr/bin/env python
# coding: utf-8

# # Model exercises

# *The content of this notebook is mainly based on Laurence Moroney's video tutorial Basic Computer Vision with ML and
# the TensorFlow tutorial "Basic classification: Classify images of clothing"

# ## Setup

# In[1]:


# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## Data preparation

# In[2]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


# ## Model

# In[3]:


model = keras.Sequential(
    [
        layers.Flatten(input_shape=(28, 28), name="layer1"),
        layers.Dense(128, activation='relu', name="layer2"),
        layers.Dropout(0.05, name="layer3"),
        layers.Dense(10, name="layer4")
])


# In[4]:


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[5]:


model.fit(train_images, train_labels, epochs=10)


# ### Evaluate accuracy

# In[6]:


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f' Test accuracy: {test_acc:.3f} \n Test loss {test_loss:.3f}')


# ## Predictions

# In[7]:


probability_model = keras.Sequential([
                        model, layers.Softmax()
                        ])

predictions = probability_model.predict(test_images)

predictions[0]


# ## Exercises

# ### Exercise 1
# 
# - For this first exercise run the below code: It creates a set of classifications for each of the test images, and then prints the first entry in the classifications. 
# - The output, after you run it is a list of numbers. 
# - Why do you think this is, and what do those numbers represent? 

# In[8]:


predictions[0].round(4)


# Hint: try running print(test_labels[0]) -- and you'll get a 9. Does that help you understand why this list looks the way it does? 

# In[9]:


print(test_labels[0])


# A: What does this list represent?
# 
# 1.   It's 10 random meaningless values
# 2.   It's the first 10 classifications that the computer made
# 3.   It's the probability that this item is each of the 10 classes

# B: How do you know that this list tells you that the item is an ankle boot?
# 
# 1.   There's not enough information to answer that question
# 1.   The 10th element on the list is the biggest, and the ankle boot is labelled 9
# 1.   The ankle boot is label 9, and there are 0->9 elements in the list
# 
# 
# 

# ### Exercise 2
# 
# - Let's now look at the layers in your model. 
# - Experiment with different values for the dense layer with a different amount of neurons. 
# - What different results do you get for loss, training time etc? Why do you think that's the case? 

# Question 1. Increase to 100 Neurons -- What's the impact?
# 
# 1. Training takes longer, but is more accurate
# 2. Training takes longer, but no impact on accuracy
# 3. Training takes the same time, but is more accurate

# ### Exercise 3
# 
# - What would happen if you remove the Flatten() layer. 
# - Why do you think that's the case? 
# 
# 
# 

# ### Exercise 4
# 
# - Consider the final (output) layers. 
# - Why are there 10 of them? 
# - What would happen if you had a different amount than 10?
# - For example, try training the network with 5

# ### Exercise 5
# 
# - Consider the effects of additional layers in the network. What will happen if you add another layer? 
# 

# ### Exercise 6
# 
# - Consider the impact of training for more or less epochs (e.g. 5 or 30). 
# - Why do you think the results may change ? 
# 
# 
