import tensorflow as tf
from models.layers.layers import variables
from models.layers.layers import _update_routing

class DigitCaps(tf.keras.layers.Layer):
  def __init__(self, hparams, num_classes, input_atoms = 8, output_atoms = 16, num_routing = 3, leaky=True):
    super(DigitCaps, self).__init__()
    self._hparams = hparams
    self.output_dim = num_classes
    self.input_atoms = input_atoms
    self.output_atoms = output_atoms
    self.num_routing = num_routing
    self.leaky = leaky

  def build(self, input_shape):
    self.input_dim = input_shape[1] 
    with tf.compat.v1.variable_scope("digit_capsule", reuse=tf.compat.v1.AUTO_REUSE):
      self.weightsW = variables.weight_variable(
          [self.input_dim, self.input_atoms, self.output_dim * self.output_atoms])
      self.biases = variables.bias_variable([self.output_dim, self.output_atoms])
    
  def call(self, inputs):
    with tf.name_scope('Wx_plus_b'):
      # Depthwise matmul: [b, d, c] ** [d, c, o_c] = [b, d, o_c]
      # To do this: tile input, do element-wise multiplication and reduce
      # sum over input_atoms dimmension.
      input_tiled = tf.tile(
          tf.expand_dims(inputs, -1),
          [1, 1, 1, self.output_dim * self.output_atoms])
      votes = tf.reduce_sum(input_tiled * self.weightsW, axis=2)
      votes_reshaped = tf.reshape(votes,
                                  [-1, self.input_dim, self.output_dim, self.output_atoms])
    with tf.name_scope('routing'):
      input_shape = tf.shape(inputs)
      logit_shape = tf.stack([tf.shape(inputs)[0], self.input_dim, self.output_dim])
      activations = _update_routing(
          votes=votes_reshaped,
          biases=self.biases,
          logit_shape=logit_shape,
          num_dims=4,
          input_dim=self.input_dim,
          output_dim=self.output_dim,
          num_routing=self.num_routing,
          leaky=self.leaky)
    return tf.norm(activations, axis=-1)