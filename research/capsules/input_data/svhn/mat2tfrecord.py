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

SVHN_FILES = {
    'train': 'train_32x32.mat',
    'test': 'test_32x32.mat'
}

SVNH_RANGE = {
    'train': (0, 73257),
    'test': (0, 26032)
}

IMAGE_SIZE_PX = 32

def int64_feature(value):
  """Casts value to a TensorFlow int64 feature list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Casts value to a TensorFlow bytes feature list."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(_):
    filename = SVHN_FILES[FLAGS.split]
    output_file = 'svhn_{}.tfrecords'.format(FLAGS.split)
    mat = scipy.io.loadmat(os.path.join(FLAGS.data_dir, filename))
    images = mat['X'].transpose(3, 2, 0, 1)
    labels = np.squeeze(mat['y'], axis=1)

    with tf.python_io.TFRecordWriter(output_file) as writer:
        for image, label in zip(images, labels):
            image_raw = image.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': int64_feature(IMAGE_SIZE_PX),
                        'width': int64_feature(IMAGE_SIZE_PX),
                        'depth': int64_feature(3),
                        'label': int64_feature(label),
                        'image_raw': bytes_feature(image_raw)
                    }
                )
            )

            writer.write(example.SerializeToString())

if __name__ == '__main__':
    tf.app.run()