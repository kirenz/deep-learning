#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.
# 

# # Load CSV data

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/load_data/csv"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/csv.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/load_data/csv.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/load_data/csv.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This tutorial provides examples of how to use CSV data with TensorFlow.
# 
# There are two main parts to this:
# 
# 1. **Loading the data off disk**
# 2. **Pre-processing it into a form suitable for training.**
# 
# This tutorial focuses on the loading, and gives some quick examples of preprocessing. For a tutorial that focuses on the preprocessing aspect see the [preprocessing layers guide](https://www.tensorflow.org/guide/keras/preprocessing_layers#quick_recipes) and [tutorial](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers). 
# 

# ## Setup

# In[1]:


import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


# ## In memory data

# For any small CSV dataset the simplest way to train a TensorFlow model on it is to load it into memory as a pandas Dataframe or a NumPy array. 
# 

# A relatively simple example is the [abalone dataset](https://archive.ics.uci.edu/ml/datasets/abalone). 
# 
# * The dataset is small. 
# * All the input features are all limited-range floating point values. 
# 
# Here is how to download the data into a [Pandas `DataFrame`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html):

# In[2]:


abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()


# The dataset contains a set of measurements of [abalone](https://en.wikipedia.org/wiki/Abalone), a type of sea snail. 
# 
# ![an abalone shell](https://tensorflow.org/images/abalone_shell.jpg)
# 
#  [“Abalone shell”](https://www.flickr.com/photos/thenickster/16641048623/) (by [Nicki Dugan Pogue](https://www.flickr.com/photos/thenickster/), CC BY-SA 2.0)
# 

# The nominal task for this dataset is to predict the age from the other measurements, so separate the features and labels for training:
# 

# In[3]:


abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')


# For this dataset you will treat all features identically. Pack the features into a single NumPy array.:

# In[4]:


abalone_features = np.array(abalone_features)
abalone_features


# Next make a regression model predict the age. Since there is only a single input tensor, a `keras.Sequential` model is sufficient here.

# In[5]:


abalone_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())


# To train that model, pass the features and labels to `Model.fit`:

# In[6]:


abalone_model.fit(abalone_features, abalone_labels, epochs=10)


# You have just seen the most basic way to train a model using CSV data. Next, you will learn how to apply preprocessing to normalize numeric columns.

# ## Basic preprocessing

# It's good practice to normalize the inputs to your model. The `experimental.preprocessing` layers provide a convenient way to build this normalization into your model. 
# 
# The layer will precompute the mean and variance of each column, and use these to normalize the data.
# 
# First you create the layer:

# In[7]:


normalize = preprocessing.Normalization()


# Then you use the `Normalization.adapt()` method to adapt the normalization layer to your data.
# 
# Note: Only use your training data to `.adapt()` preprocessing layers. Do not use your validation or test data.

# In[8]:


normalize.adapt(abalone_features)


# Then use the normalization layer in your model:

# In[9]:


norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.losses.MeanSquaredError(),
                           optimizer = tf.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)


# ## Mixed data types
# 
# The "Titanic" dataset contains information about the passengers on the Titanic. The nominal task on this dataset is to predict who survived. 
# 
# ![The Titanic](img/Titanic.jpg)
# 
# Image [from Wikimedia](https://commons.wikimedia.org/wiki/File:RMS_Titanic_3.jpg)
# 
# The raw data can easily be loaded as a Pandas `DataFrame`, but is not immediately usable as input to a TensorFlow model. 
# 

# In[10]:


titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()


# In[11]:


titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')


# Because of the different data types and ranges you can't simply stack the features into  NumPy array and pass it to a `keras.Sequential` model. Each column needs to be handled individually. 
# 
# As one option, you could preprocess your data offline (using any tool you like) to convert categorical columns to numeric columns, then pass the processed output to your TensorFlow model. The disadvantage to that approach is that if you save and export your model the preprocessing is not saved with it. The `experimental.preprocessing` layers avoid this problem because they're part of the model.
# 

# In this example, you'll build a model that implements the preprocessing logic using [Keras functional API](https://www.tensorflow.org/guide/keras/functional). You could also do it by [subclassing](https://www.tensorflow.org/guide/keras/custom_layers_and_models).
# 
# The functional API operates on "symbolic" tensors. Normal "eager" tensors have a value. In contrast these "symbolic" tensors do not. Instead they keep track of which operations are run on them, and build representation of the calculation, that you can run later. Here's a quick example:

# In[12]:


# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Do a calculation using is
result = 2*input + 1

# the result doesn't have a value
result


# In[13]:


calc = tf.keras.Model(inputs=input, outputs=result)


# In[14]:


print(calc(1).numpy())
print(calc(2).numpy())


# To build the preprocessing model, start by building a set of symbolic `keras.Input` objects, matching the names and data-types of the CSV columns.

# In[15]:


inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs


# The first step in your preprocessing logic is to concatenate the numeric inputs together, and run them through a normalization layer:

# In[16]:


numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs


# Collect all the symbolic preprocessing results, to concatenate them later.

# In[17]:


preprocessed_inputs = [all_numeric_inputs]


# For the string inputs use the `preprocessing.StringLookup` function to map from strings to integer indices in a vocabulary. Next, use `preprocessing.CategoryEncoding` to convert the indexes into `float32` data appropriate for the model. 
# 
# The default settings for the `preprocessing.CategoryEncoding` layer create a one-hot vector for each input. A `layers.Embedding` would also work. See the [preprocessing layers guide](https://www.tensorflow.org/guide/keras/preprocessing_layers#quick_recipes) and [tutorial](../structured_data/preprocessing_layers.ipynb) for more on this topic.

# In[18]:


for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue
  
  lookup = preprocessing.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)


# With the collection of `inputs` and `processed_inputs`, you can concatenate all the preprocessed inputs together, and build a model that handles the preprocessing:

# In[22]:


get_ipython().system('conda install --yes -c anaconda pydot')


# In[23]:


preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)


# This `model` just contains the input preprocessing. You can run it to see what it does to your data. Keras models don't automatically convert Pandas `DataFrames` because it's not clear if it should be converted to one tensor or to a dictionary of tensors. So convert it to a dictionary of tensors:

# In[24]:


titanic_features_dict = {name: np.array(value) 
                         for name, value in titanic_features.items()}


# Slice out the first training example and pass it to this preprocessing model, you see the numeric features and string one-hots all concatenated together:

# In[25]:


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)


# Now build the model on top of this:

# In[26]:


def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)


# When you train the model, pass the dictionary of features as `x`, and the label as `y`.

# In[27]:


titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)


# Since the preprocessing is part of the model, you can save the model and reload it somewhere else and get identical results:

# In[28]:


titanic_model.save('test')
reloaded = tf.keras.models.load_model('test')


# In[29]:


features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)


# ## Using tf.data
# 

# In the previous section you relied on the model's built-in data shuffling and batching while training the model. 
# 
# If you need more control over the input data pipeline or need to use data that doesn't easily fit into memory: use `tf.data`. 
# 
# For more examples see the [tf.data guide](../../guide/data.ipynb).

# ### On in memory data
# 
# As a first example of applying `tf.data` to CSV data consider the following code to manually slice up the dictionary of features from the previous section. For each index, it takes that index for each feature:
# 

# In[30]:


import itertools

def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example


# Run this and print the first example:

# In[31]:


for example in slices(titanic_features_dict):
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break


# The most basic `tf.data.Dataset` in memory data loader is the `Dataset.from_tensor_slices` constructor. This returns a `tf.data.Dataset` that implements a generalized version of the above `slices` function, in TensorFlow. 

# In[32]:


features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)


# You can iterate over a `tf.data.Dataset` like any other python iterable:

# In[33]:


for example in features_ds:
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break


# The `from_tensor_slices` function can handle any structure of nested dictionaries or tuples. The following code makes a dataset of `(features_dict, labels)` pairs:

# In[34]:


titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))


# To train a model using this `Dataset`, you'll need to at least `shuffle` and `batch` the data.

# In[35]:


titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)


# Instead of passing `features` and `labels` to `Model.fit`, you pass the dataset:

# In[36]:


titanic_model.fit(titanic_batches, epochs=5)


# ### From a single file
# 
# So far this tutorial has worked with in-memory data. `tf.data` is a highly scalable toolkit for building data pipelines, and provides a few functions for dealing loading CSV files. 

# In[34]:


titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")


# Now read the CSV data from the file and create a `tf.data.Dataset`. 
# 
# (For the full documentation, see `tf.data.experimental.make_csv_dataset`)
# 

# In[35]:


titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5, # Artificially small to make examples easier to show.
    label_name='survived',
    num_epochs=1,
    ignore_errors=True,)


# This function includes many convenient features so the data is easy to work with. This includes:
# 
# * Using the column headers as dictionary keys.
# * Automatically determining the type of each column.

# In[36]:


for batch, label in titanic_csv_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value}")
  print()
  print(f"{'label':20s}: {label}")


# Note: if you run the above cell twice it will produce different results. The default settings for `make_csv_dataset` include `shuffle_buffer_size=1000`, which is more than sufficient for this small dataset, but may not be for a real-world dataset.

# It can also decompress the data on the fly. Here's a gzipped CSV file containing the [metro interstate traffic dataset](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)
# 
# ![A traffic jam.](images/csv/traffic.jpg)
# 
# Image [from Wikimedia](https://commons.wikimedia.org/wiki/File:Trafficjam.jpg)
# 

# In[37]:


traffic_volume_csv_gz = tf.keras.utils.get_file(
    'Metro_Interstate_Traffic_Volume.csv.gz', 
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz",
    cache_dir='.', cache_subdir='traffic')


# Set the `compression_type` argument to read directly from the compressed file: 

# In[38]:


traffic_volume_csv_gz_ds = tf.data.experimental.make_csv_dataset(
    traffic_volume_csv_gz,
    batch_size=256,
    label_name='traffic_volume',
    num_epochs=1,
    compression_type="GZIP")

for batch, label in traffic_volume_csv_gz_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value[:5]}")
  print()
  print(f"{'label':20s}: {label[:5]}")


# Note: If you need to parse those date-time strings in the `tf.data` pipeline you can use `tfa.text.parse_time`.

# ### Caching

# There is some overhead to parsing the csv data. For small models this can be the bottleneck in training.
# 
# Depending on your use case it may be a good idea to use `Dataset.cache` or `data.experimental.snapshot` so that the csv data is only parsed on the first epoch. 
# 
# The main difference between the `cache` and `snapshot` methods is that `cache` files can only be used by the TensorFlow process that created them, but `snapshot` files can be read by other processes.
# 
# For example, iterating over the `traffic_volume_csv_gz_ds` 20 times, takes ~15 seconds without caching, or ~2s with caching.

# In[39]:


get_ipython().run_cell_magic('time', '', "for i, (batch, label) in enumerate(traffic_volume_csv_gz_ds.repeat(20)):\n  if i % 40 == 0:\n    print('.', end='')\nprint()")


# Note: `Dataset.cache`  stores the data form the first epoch and replays it in order. So using `.cache` disables any shuffles earlier in the pipeline. Below the `.shuffle` is added back in after `.cache`.

# In[40]:


get_ipython().run_cell_magic('time', '', "caching = traffic_volume_csv_gz_ds.cache().shuffle(1000)\n\nfor i, (batch, label) in enumerate(caching.shuffle(1000).repeat(20)):\n  if i % 40 == 0:\n    print('.', end='')\nprint()")


# Note: `snapshot` files are meant for *temporary* storage of a dataset while in use. This is *not* a format for long term storage. The file format is considered an internal detail, and not guaranteed between TensorFlow versions. 

# In[41]:


get_ipython().run_cell_magic('time', '', "snapshot = tf.data.experimental.snapshot('titanic.tfsnap')\nsnapshotting = traffic_volume_csv_gz_ds.apply(snapshot).shuffle(1000)\n\nfor i, (batch, label) in enumerate(snapshotting.shuffle(1000).repeat(20)):\n  if i % 40 == 0:\n    print('.', end='')\nprint()")


# If your data loading is slowed by loading csv files, and `cache` and `snapshot` are insufficient for your use case, consider re-encoding your data into a more streamlined format.

# ### Multiple files

# All the examples so far in this section could easily be done without `tf.data`. One place where `tf.data` can really simplify things is when dealing with collections of files.
# 
# For example, the [character font images](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images) dataset is distributed as a collection of csv files, one per font.
# 
# ![Fonts](images/csv/fonts.jpg)
# 
# Image by <a href="https://pixabay.com/users/wilhei-883152/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=705667">Willi Heidelbach</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=705667">Pixabay</a>
# 
# Download the dataset, and have a look at the files inside:

# In[42]:


fonts_zip = tf.keras.utils.get_file(
    'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True)


# In[43]:


import pathlib
font_csvs =  sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

font_csvs[:10]


# In[44]:


len(font_csvs)


# When dealing with a bunch of files you can pass a glob-style `file_pattern` to the `experimental.make_csv_dataset` function. The order of the files is shuffled each iteration.
# 
# Use the `num_parallel_reads` argument to set how many files are read in parallel and interleaved together.

# In[45]:


fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)


# These csv files have the images flattened out into a single row. The column names are formatted `r{row}c{column}`. Here's the first batch:

# In[46]:


for features in fonts_ds.take(1):
  for i, (name, value) in enumerate(features.items()):
    if i>15:
      break
    print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")


# #### Optional: Packing fields
# 
# You probably don't want to work with each pixel in separate columns like this. Before trying to use this dataset be sure to pack the pixels into an image-tensor. 
# 
# Here is code that parses the column names to build images for each example:

# In[47]:


import re

def make_images(features):
  image = [None]*400
  new_feats = {}

  for name, value in features.items():
    match = re.match('r(\d+)c(\d+)', name)
    if match:
      image[int(match.group(1))*20+int(match.group(2))] = value
    else:
      new_feats[name] = value

  image = tf.stack(image, axis=0)
  image = tf.reshape(image, [20, 20, -1])
  new_feats['image'] = image

  return new_feats


# Apply that function to each batch in the dataset:

# In[48]:


fonts_image_ds = fonts_ds.map(make_images)

for features in fonts_image_ds.take(1):
  break


# Plot the resulting images:

# In[49]:


from matplotlib import pyplot as plt

plt.figure(figsize=(6,6), dpi=120)

for n in range(9):
  plt.subplot(3,3,n+1)
  plt.imshow(features['image'][..., n])
  plt.title(chr(features['m_label'][n]))
  plt.axis('off')


# ## Lower level functions

# So far this tutorial has focused on the highest level utilities for reading csv data. There are other two APIs that may be helpful for advanced users if your use-case doesn't fit the basic patterns.
# 
# * `tf.io.decode_csv` - a function for parsing lines of text into a list of CSV column tensors.
# * `tf.data.experimental.CsvDataset` - a lower level csv dataset constructor.
# 
# This section recreates functionality provided by `make_csv_dataset`, to demonstrate how this lower level functionality can be used.
# 

# ### `tf.io.decode_csv`
# 
# This function decodes a string, or list of strings into a list of columns.
# 
# Unlike `make_csv_dataset` this function does not try to guess column data-types. You specify the column types by providing a list of `record_defaults` containing a value of the correct type, for each column.
# 
# To read the Titanic data **as strings** using `decode_csv` you would say: 

# In[50]:


text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]

all_strings = [str()]*10
all_strings


# In[51]:


features = tf.io.decode_csv(lines, record_defaults=all_strings) 

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")


# To parse them with their actual types, create a list of `record_defaults` of the corresponding types: 

# In[52]:


print(lines[0])


# In[53]:


titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types


# In[54]:


features = tf.io.decode_csv(lines, record_defaults=titanic_types) 

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")


# Note: it is more efficient to call `decode_csv` on large batches of lines than on individual lines of csv text.

# ### `tf.data.experimental.CsvDataset`
# 
# The `tf.data.experimental.CsvDataset` class provides a minimal CSV `Dataset` interface without the convenience features of the `make_csv_dataset` function: column header parsing, column type-inference, automatic shuffling, file interleaving.
# 
# This constructor follows uses `record_defaults` the same way as `io.parse_csv`:
# 

# In[55]:


simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)

for example in simple_titanic.take(1):
  print([e.numpy() for e in example])


# The above code is basically equivalent to:

# In[56]:


def decode_titanic_line(line):
  return tf.io.decode_csv(line, titanic_types)

manual_titanic = (
    # Load the lines of text
    tf.data.TextLineDataset(titanic_file_path)
    # Skip the header row.
    .skip(1)
    # Decode the line.
    .map(decode_titanic_line)
)

for example in manual_titanic.take(1):
  print([e.numpy() for e in example])


# #### Multiple files
# 
# To parse the fonts dataset using `experimental.CsvDataset`, you first need to determine the column types for the `record_defaults`. Start by inspecting the first row of one file: 

# In[57]:


font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)


# Only the first two fields are strings, the rest are ints or floats, and you can get the total number of features by counting the commas:

# In[58]:


num_font_features = font_line.count(',')+1
font_column_types = [str(), str()] + [float()]*(num_font_features-2)


# The `CsvDatasaet` constructor can take a list of input files, but reads them sequentially. The first file in the list of CSVs is `AGENCY.csv`:

# In[59]:


font_csvs[0]


# So when you pass pass the list of files to `CsvDataaset` the records from `AGENCY.csv` are read first:

# In[60]:


simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs, 
    record_defaults=font_column_types, 
    header=True)


# In[61]:


for row in simple_font_ds.take(10):
  print(row[0].numpy())


# To interleave multiple files, use `Dataset.interleave`.
# 
# Here's an initial dataset that contains the csv file names: 

# In[62]:


font_files = tf.data.Dataset.list_files("fonts/*.csv")


# This shuffles the file names each epoch:

# In[63]:


print('Epoch 1:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
  print("    ", f.numpy())
print('    ...')


# The `interleave` method takes a `map_func` that creates a child-`Dataset` for each element of the parent-`Dataset`. 
# 
# Here, you want to create a `CsvDataset` from each element of the dataset of files:

# In[64]:


def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
    path, 
    record_defaults=font_column_types, 
    header=True)


# The `Dataset` returned by interleave returns elements by cycling over a number of the child-`Dataset`s. Note, below, how the dataset cycles over `cycle_length)=3` three font files:

# In[65]:


font_rows = font_files.interleave(make_font_csv_ds,
                                  cycle_length=3)


# In[66]:


fonts_dict = {'font_name':[], 'character':[]}

for row in font_rows.take(10):
  fonts_dict['font_name'].append(row[0].numpy().decode())
  fonts_dict['character'].append(chr(row[2].numpy()))

pd.DataFrame(fonts_dict)


# #### Performance
# 

# Earlier, it was noted that `io.decode_csv` is more efficient when run on a batch of strings.
# 
# It is possible to take advantage of this fact, when using large batch sizes, to improve CSV loading performance (but try [caching](#caching) first).

# With the built-in loader 20, 2048-example batches take about 17s. 

# In[67]:


BATCH_SIZE=2048
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=BATCH_SIZE, num_epochs=1,
    num_parallel_reads=100)


# In[68]:


get_ipython().run_cell_magic('time', '', "for i,batch in enumerate(fonts_ds.take(20)):\n  print('.',end='')\n\nprint()")


# Passing **batches of text lines** to`decode_csv` runs faster, in about 5s:

# In[69]:


fonts_files = tf.data.Dataset.list_files("fonts/*.csv")
fonts_lines = fonts_files.interleave(
    lambda fname:tf.data.TextLineDataset(fname).skip(1), 
    cycle_length=100).batch(BATCH_SIZE)

fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))


# In[70]:


get_ipython().run_cell_magic('time', '', "for i,batch in enumerate(fonts_fast.take(20)):\n  print('.',end='')\n\nprint()")


# For another example of increasing csv performance by using large batches see the [overfit and underfit tutorial](../keras/overfit_and_underfit.ipynb).
# 
# This sort of approach may work, but consider other options like `cache` and `snapshot`, or re-enncoding your data into a more streamlined format.
