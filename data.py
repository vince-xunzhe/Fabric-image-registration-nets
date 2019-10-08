import numpy as np
import cv2
import os
from pathlib import Path

class AryllaDataHandler(object):
  
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()
  
  def _get_data(self):

    if self.is_train:
      Image_path = 'C:/Users/xwen2/Desktop/DIRNet/DataProcessed/Training/'
      Label_path = 'C:/Users/xwen2/Desktop/DIRNet/Label/training_label.txt'
    else :
      Image_path = 'C:/Users/xwen2/Desktop/DIRNet/DataProcessed/Testing/'
      Label_path = 'C:/Users/xwen2/Desktop/DIRNet/Label/testing_label.txt'
    
    images = []  # ndarray
    labels = []

    for filename in os.listdir(Image_path):
      img = cv2.imread(Image_path + filename)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
      images.append(img)
    images = np.asarray(images)
    images = np.expand_dims(images, axis=3)

    f = open(Label_path, 'r')
    lines = f.readlines()
    for line in lines:
      if line is not "":
        line = line.split(',')
        labels.append(line[1])
    f.close()
    labels = np.asarray(labels, dtype=np.uint8)

    values, counts = np.unique(labels, return_counts=True)

    data = []
    
    for i in range(2):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 220, 200, 1], dtype=np.float32)
      data.append(arr)

    l_iter = [0]*2

    for i in range(labels.shape[0]):
      label = labels[i]
      
      data[label][l_iter[label]] = images[i] / 255.
      l_iter[label] += 1

    return data

  def sample_pair(self, batch_size, label=None):
    
    label = np.random.randint(2) if label is None else label
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y




class MNISTDataHandler(object):
  
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()

  def _get_data(self):
    from tensorflow.contrib.learn.python.learn.datasets.base \
      import maybe_download
    from tensorflow.contrib.learn.python.learn.datasets.mnist \
      import extract_images, extract_labels

    if self.is_train:
      IMAGES = 'train-images-idx3-ubyte.gz'
      LABELS = 'train-labels-idx1-ubyte.gz'
    else :
      IMAGES = 't10k-images-idx3-ubyte.gz'
      LABELS = 't10k-labels-idx1-ubyte.gz'

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    local_file = maybe_download(IMAGES, self.path, SOURCE_URL)

    with open(local_file, 'rb') as f:
      images = extract_images(f)
    local_file = maybe_download(LABELS, self.path, SOURCE_URL)
    with open(local_file, 'rb') as f:
      labels = extract_labels(f, one_hot=False)

    values, counts = np.unique(labels, return_counts=True)

    data = []
    
    for i in range(10):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 28, 28, 1], dtype=np.float32)
      data.append(arr)

    l_iter = [0]*10

    for i in range(labels.shape[0]):
      label = labels[i]
      data[label][l_iter[label]] = images[i] / 255.
      l_iter[label] += 1

    return data

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(10) if label is None else label
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y
