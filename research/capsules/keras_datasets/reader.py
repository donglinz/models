import tensorflow as tf
import tensorflow_datasets as tfds
def read_cifar10(epoches, batch_size):
  (ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
    shuffle_files=True
  )

  image_size = 24
  def normalize_img_train(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(
        image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, [2, 0, 1])
    return image, label

  def normalize_img_test(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    image = tf.image.per_image_standardization(image)
    image = tf.transpose(image, [2, 0, 1])
    return image, label

  ds_train = ds_train.cache().repeat(epoches).map(normalize_img_train).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
  ds_test = ds_test.cache().repeat(epoches).map(normalize_img_test).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

  return ds_train, ds_test
