import tensorflow as tf
from keras_models.conv_capsule import ConvCapsule
from keras_models.digit_caps import DigitCaps
import keras_datasets.reader
from datetime import datetime

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('data_dir', None, 'The data directory.')
tf.compat.v1.flags.DEFINE_integer('eval_size', 10000, 'Size of the test dataset.')
tf.compat.v1.flags.DEFINE_string('hparams_override', None,
                       'A string of form key=value,key=value to override the'
                       'hparams of this experiment.')
tf.compat.v1.flags.DEFINE_integer('max_steps', 1000, 'Number of steps to train.')
tf.compat.v1.flags.DEFINE_string('model', 'capsule',
                       'The model to use for the experiment.'
                       'capsule or baseline')
tf.compat.v1.flags.DEFINE_string('dataset', 'mnist',
                       'The dataset to use for the experiment.'
                       'mnist, norb, cifar10.')
tf.compat.v1.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus to use.')
tf.compat.v1.flags.DEFINE_integer('num_targets', 1,
                        'Number of targets to detect (1 or 2).')
tf.compat.v1.flags.DEFINE_integer('num_trials', 1,
                        'Number of trials for ensemble evaluation.')
tf.compat.v1.flags.DEFINE_integer('save_step', 1500, 'How often to save checkpoints.')
tf.compat.v1.flags.DEFINE_string('summary_dir', None,
                       'Main directory for the experiments.')
tf.compat.v1.flags.DEFINE_string('checkpoint', None,
                       'The model checkpoint for evaluation.')
tf.compat.v1.flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
tf.compat.v1.flags.DEFINE_bool('validate', False, 'Run trianing/eval in validation mode.')

import ctypes

_cudart = ctypes.CDLL('libcudart.so')
def cu_prof_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception('cudaProfilerStart() returned %d' % ret)

def cu_prof_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception('cudaProfilerStop() returned %d' % ret)

class ProfileCallback(tf.keras.callbacks.Callback):
  def __init__(self, profile_batch):
    super().__init__()
    self.profile_batch = profile_batch
  def on_train_batch_begin(self, batch, logs=None):
    if batch == self.profile_batch:
      cu_prof_start()
    if batch == self.profile_batch + 1:
      cu_prof_stop()

class Hparams():
  def __init__(self):
    self.decay_rate=0.96
    self.decay_steps=2000
    self.leaky=True
    self.learning_rate=0.001
    self.loss_type='margin'
    self.num_prime_capsules=64
    self.padding='SAME'
    self.remake=False
    self.routing=3
    self.verbose=False

def keras_train(hparams):
  num_classes = 10
  epochs = 2
  batch_size = 200
  #model = keras_models.capsule_model.CapsuleModel(hparams, num_classes, batch_size)
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
      256, (9, 9), 
      activation='relu',
      padding = hparams.padding,
      data_format = 'channels_first',
      input_shape=(3, 24, 24)),
    ConvCapsule(
      hparams = hparams,
      input_dim = 1,
      output_dim = hparams.num_prime_capsules,
      input_atoms = 256, 
      output_atoms = 8, 
      stride = 2, 
      kernel_size = 9,
      num_routing = 1,
      leaky = hparams.leaky
    ),
    DigitCaps(
      hparams = hparams,
      num_classes = num_classes, 
      input_atoms = 8, 
      output_atoms = 16,
      num_routing = hparams.routing,
      leaky = hparams.leaky)
  ])

  train, test = keras_datasets.reader.read_cifar10(epochs, batch_size)

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])


  # Create a TensorBoard callback
  logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

  # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
  #                                                 histogram_freq = 1,
  #                                                 profile_batch = '100,110')
  model.summary()
  model.fit(train,
          epochs=epochs,
          validation_data=test,
          steps_per_epoch=50000/batch_size,
          validation_steps=10000/batch_size)



keras_train(Hparams())
