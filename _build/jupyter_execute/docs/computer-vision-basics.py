#!/usr/bin/env python
# coding: utf-8

# # Computer vision - build a neural network
# 
# *The content of this notebook is based on the official TensorFlow tutorial "TensorFlow 2 quickstart for beginners". Here is the original source code:*

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/quickstart/beginner"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/quickstart/beginner.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This short introduction uses [Keras](https://www.tensorflow.org/guide/keras/overview) to:
# 
# 1. Build a neural network that classifies images.
# 2. Train this neural network.
# 3. And, finally, evaluate the accuracy of the model.

# ## Data preparation

# In[3]:


import tensorflow as tf


# Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:

# In[4]:


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# ## Neural network design

# Build the `tf.keras.Sequential` model by stacking layers. The ``Dropout`` layer randomly sets input units to 0 with a frequency of rate 0.2 at each step during training time, which helps prevent overfitting. 

# In[5]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])


# ### Optimizer and loss function

# Next, we need to choose an **optimizer** and **loss** function for training:

# For each example the model returns a vector of "[logits](https://developers.google.com/machine-learning/glossary#logits)" or "[log-odds](https://developers.google.com/machine-learning/glossary#log-odds)" scores, one for each class.

# In[6]:


predictions = model(x_train[:1]).numpy()
predictions


# The `tf.nn.softmax` function converts these logits to "probabilities" for each class: 

# In[7]:


tf.nn.softmax(predictions).numpy()


# Note: It is possible to bake this `tf.nn.softmax` in as the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to
# provide an exact and numerically stable loss calculation for all models when using a softmax output. 

# The `losses.SparseCategoricalCrossentropy` loss takes a vector of logits and a `True` index and returns a scalar loss for each example.

# In[8]:


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# This loss is equal to the negative log probability of the true class:
# It is zero if the model is sure of the correct class.
# 
# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.math.log(1/10) ~= 2.3`.

# In[10]:


loss_fn(y_train[:1], predictions).numpy()


# In[12]:


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


# ### Model training

# The `Model.fit` method adjusts the model parameters to minimize the loss: 

# In[13]:


model.fit(x_train, y_train, epochs=5)


# ## Model evaluation
# 
# The `Model.evaluate` method checks the models performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)".

# In[15]:


model.evaluate(x_test,  y_test, verbose=2)


# The image classifier is now trained to ~98% accuracy on this dataset.

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

# In[16]:


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


# In[17]:


probability_model(x_test[:5])

