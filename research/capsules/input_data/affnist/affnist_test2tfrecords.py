import tensorflow as tf
import numpy as np
import os
import scipy.io

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data',
                       'Directory for storing input data')
IMAGE_SIZE_PX = 40

def int64_feature(value):
  """Casts value to a TensorFlow int64 feature list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Casts value to a TensorFlow bytes feature list."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(_):
  filename = os.path.join(FLAGS.data_dir, 'test.mat')
  outfile = 'affnist_test.tfrecords'
  mat = scipy.io.loadmat(filename)
  images = mat['affNISTdata'][0][0][2].transpose(1, 0)
  lables = mat['affNISTdata'][0][0][5].transpose(1, 0)

  with tf.python_io.TFRecordWriter(outfile) as writer:
    for image, label in zip(images, lables):
          image_raw = image.tostring()
          example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'height': int64_feature(IMAGE_SIZE_PX),
                      'width': int64_feature(IMAGE_SIZE_PX),
                      'depth': int64_feature(1),
                      'label': int64_feature(label),
                      'image_raw': bytes_feature(image_raw),
                  }))
          writer.write(example.SerializeToString())
    
if __name__ == '__main__':
  tf.app.run()