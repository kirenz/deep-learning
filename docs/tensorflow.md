# TensorFlow

## Installation

[TensorFlow](https://www.tensorflow.org) is an open source platform for machine learning provided by Google. In this course, we will use TensorFlow 2 in combination with [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx) -which is an end-to-end platform for deploying production ML pipelines- to build deep neural networks. 


```{admonition} Installation tutorial
:class: tip
- [Installation of TFX](https://kirenz.github.io/codelabs/codelabs/tfx-install/#0)
```

## Keras

Built on top of TensorFlow 2, [Keras](https://keras.io) is an industry-strength framework that can scale to large clusters of GPUs or an entire TPU pod. It is a central part of the tightly-connected TensorFlow 2 ecosystem, covering every step of the machine learning workflow, from data management to hyperparameter training to deployment solutions.

Keras is used by CERN (and yes, Keras is used at the [LHC](https://blog.tensorflow.org/2021/04/reconstructing-thousands-of-particles-in-one-go-at-cern-lhc.html)), NASA and many more scientific organizations around the world. Furthermore, it is the most used deep learning framework among top-5 winning teams on [Kaggle](https://www.kaggle.com). 

Note that Keras offers many [code examples](https://keras.io/examples/) with short (less than 300 lines of code), focused demonstrations of deep learning workflows. All of the examples are written as Jupyter notebooks and can be run in one click in Google Colab.

## First steps

Next, we take a look at how to build a deep neural network model using TensorFlow 2 and Keras. The content is based on Laurence Moroney's excellent Tutorial "Intro to Machine Learning" (see video below): 

:::{Note}
:::

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQn-aJbtwbwSJgA9mMMRBUXrtIvKJXQWwNrMpAr4tPKZ1URiI84eWvlZPly3wcDpp0e6NrrbGVW5G2g/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="520" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Resources
:class: tip
- [Download slides](https://docs.google.com/presentation/d/17paB63OQX_GU7ldbRwduYevAHWqJVGJkV4NZdXPpKjY/export/pdf)
- [Jupyter Notebook](https://kirenz.github.io/deep-learning/docs/tf-example.html)

```

Google'S AI Advocate Laurence Moroney walks you through the code provided in the presentation: 

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/KNAWp2S3w94" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>