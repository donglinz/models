import scipy.io as sio
from glob import glob
import numpy as np
import tensorflow as tf
import math
import os 

def load_data_from_mat(path):
	data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
	for key in data:
		if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
			data[key] = _todict(data[key])
	return data

def one_hot(label, output_dim):
	one_hot = np.zeros((len(label), output_dim))
	
	for idx in range(0,len(label)):
		one_hot[idx, label[idx]] = 1
	
	return one_hot

def _todict(matobj):
  #A recursive function which constructs from matobjects nested dictionaries
  dict = {}
  for strg in matobj._fieldnames:
    elem = matobj.__dict__[strg]
    if isinstance(elem, sio.matlab.mio5_params.mat_struct):
      dict[strg] = _todict(elem)
    else:
      dict[strg] = elem
  return dict

def read_affnist(split, batch_size, path):
  train_path = glob(os.path.join(path, "train/*.mat"))
  test_path = glob(os.path.join(path, "test/*.mat"))

  train_data = load_data_from_mat(train_path[0])

  trainX = train_data['affNISTdata']['image'].transpose()
  trainY = train_data['affNISTdata']['label_int']

  trainX = trainX.reshape((50000, 40, 40, 1)).astype(np.float32)
  trainY = trainY.reshape((50000)).astype(np.int32)
  trainY = one_hot(trainY, 10)

  test_data = load_data_from_mat(test_path[0])
  testX = test_data['affNISTdata']['image'].transpose()
  testY = test_data['affNISTdata']['label_int']

  testX = testX.reshape((10000, 40, 40, 1)).astype(np.float32)
  testY = testY.reshape((10000)).astype(np.int32)
  testY_one_hot = one_hot(testY, 10)	

  if split == "train":
    X = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255.
    Y = tf.convert_to_tensor(trainY, dtype=tf.float32)
  elif split == "test":
    X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
    Y = tf.convert_to_tensor(testY_one_hot, dtype=tf.float32)
  else:
    raise Exception("not implemented.")

  input_queue = tf.train.slice_input_producer([X, Y],shuffle=True)
  images = tf.image.resize_images(input_queue[0] ,[40, 40])
  labels = input_queue[1]

  X, Y = tf.train.batch([images, labels],
						  batch_size=batch_size
						  )
  
  batched_features = {}
  batched_features['height'] = 40
  batched_features['depth'] = 1
  batched_features['num_targets'] = 1
  batched_features['num_classes'] = 10
  batched_features['images'] = tf.transpose(X, [0, 3, 1, 2])
  batched_features['labels'] = Y
  batched_features['recons_image'] = tf.transpose(X, [0, 3, 1, 2])
  batched_features['recons_label'] = testY
  return batched_features
