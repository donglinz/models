import scipy.io
import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data',
                       'Directory for storing input data')
tf.flags.DEFINE_string(
    'split', 'train',
    'The split of data to process: train, test, valid_train or valid_test.')

NORB_FILES = {
    'train': ('smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat'),
    'test': ('smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat', 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')
}

NORB_RANGE = {
    'train': (0, 24300),
    'test': (0, 24300)
}

IMAGE_SIZE_PX = 96

def int64_feature(value):
  """Casts value to a TensorFlow int64 feature list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Casts value to a TensorFlow bytes feature list."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_file(file_bytes, header_byte_size, data_size,dtype=np.uint8):
  """Discards 4 * header_byte_size of file_bytes and returns data_size bytes."""
  file_bytes.read(4 * header_byte_size)
  return np.frombuffer(file_bytes.read(data_size), dtype=dtype)

def read_byte_data(data_dir, split):
  """Extracts images and labels from MNIST binary file.

  Reads the binary image and label files for the given split. Generates a
  tuple of numpy array containing the pairs of label and image.
  The format of the binary files are defined at:
    http://yann.lecun.com/exdb/mnist/
  In summary: header size for image files is 4 * 4 bytes and for label file is
  2 * 4 bytes.

  Args:
    data_dir: String, the directory containing the dataset files.
    split: String, the dataset split to process. It can be one of train, test,
      valid_train, valid_test.
  Returns:
    A list of (image, label). Image is a 28x28 numpy array and label is an int.
  """
  image_file, label_file = (
      os.path.join(data_dir, file_name) for file_name in NORB_FILES[split])
  start, end = NORB_RANGE[split]
  with open(image_file, 'rb') as f:
    images = read_file(f, 6, end * 2 * IMAGE_SIZE_PX * IMAGE_SIZE_PX)
    images = images.reshape(end, 2, IMAGE_SIZE_PX, IMAGE_SIZE_PX)
  with open(label_file, 'rb') as f:
    labels = read_file(f, 5, end*4, dtype=np.uint32)

  return zip(images[start:], labels[start:])

def main(_):
    output_file = '{}duo.tfrecords'.format(FLAGS.split)
    dataset = read_byte_data(FLAGS.data_dir, FLAGS.split)
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for image, label in dataset:
            image_raw = image.tostring()
            example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height': int64_feature(IMAGE_SIZE_PX),
                    'width': int64_feature(IMAGE_SIZE_PX),
                    'depth': int64_feature(2),
                    'label': int64_feature(label),
                    'image_raw': bytes_feature(image_raw),
                }))
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    tf.app.run()