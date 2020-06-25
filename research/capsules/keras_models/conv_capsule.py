import tensorflow as tf
from models.layers import variables
from models.layers.layers import _update_routing

class ConvCapsule(tf.keras.layers.Layer):
  def __init__(self, hparams, input_dim, output_dim, input_atoms, output_atoms, stride, kernel_size, num_routing=1, leaky=True):
    super(ConvCapsule, self).__init__()
    self.hparams = hparams
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.input_atoms = input_atoms
    self.output_atoms = output_atoms
    self.stride = stride
    self.kernel_size = kernel_size
    self.num_routing = num_routing
    self.leaky = leaky
  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv2D(
      filters = self.output_dim * self.output_atoms, 
      kernel_size = self.kernel_size,
      strides = self.stride,
      padding = self.hparams.padding,
      data_format = 'channels_first')
    self._input_shape = input_shape
    with tf.compat.v1.variable_scope("conv_capsule", reuse=tf.compat.v1.AUTO_REUSE):
      self.biases = variables.bias_variable([self.output_dim, self.output_atoms, 1, 1])

    #self.input_shape = input_shape
    _, _, self.in_height, self.in_width = input_shape
  def call(self, inputs):
    ret = self.conv(inputs)
    ret_shape = tf.shape(ret)
    _, _, ret_height, ret_width = ret.get_shape()

    ret_reshaped = tf.reshape(ret, [
        -1, self.input_dim, self.output_dim, self.output_atoms, ret_shape[2], ret_shape[3]
    ])

    ret_reshaped.set_shape((None, self.input_dim, self.output_dim, self.output_atoms,
                             ret_height, ret_width))

    with tf.name_scope('routing'):
      logit_shape = tf.stack([
          tf.shape(inputs)[0], self.input_dim, self.output_dim, ret_shape[2], ret_shape[3]
      ])
      biases_replicated = tf.tile(self.biases,
                                  [1, 1, ret_shape[2], ret_shape[3]])
      activations = _update_routing(
          votes=ret_reshaped,
          biases=biases_replicated,
          logit_shape=logit_shape,
          num_dims=6,
          input_dim=self.input_dim,
          output_dim=self.output_dim,
          num_routing=self.num_routing,
          leaky=self.leaky)

      act_reshaped = tf.transpose(activations, [0, 1, 3, 4, 2])
      capsule1_3d = tf.reshape(act_reshaped,
                             [tf.shape(inputs)[0], -1, 8])
      capsule1_3d.set_shape((None, act_reshaped.shape[1] * act_reshaped.shape[2] * act_reshaped.shape[3], 8))

    return capsule1_3d
    

