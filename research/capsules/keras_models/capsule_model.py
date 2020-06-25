import tensorflow as tf
from keras_models.conv_capsule import ConvCapsule
from keras_models.digit_caps import DigitCaps

class CapsuleModel(tf.keras.Model):
  def __init__(self, hparams, num_classes, batch_size):
    super(CapsuleModel, self).__init__()
    self._hparams = hparams
    self.num_classes = num_classes
    self.batch_size = batch_size
    
    
  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv2D(
      256, (9, 9), 
      activation='relu',
      padding = self._hparams.padding,
      data_format = 'channels_first',
      input_shape=(3, 24, 24))
    
    self.conv_caps = ConvCapsule(
      hparams = self._hparams,
      input_dim = 1,
      output_dim = self._hparams.num_prime_capsules,
      input_atoms = 256, 
      output_atoms = 8, 
      stride = 2, 
      kernel_size = 9,
      num_routing = 1,
      leaky = self._hparams.leaky
    )

    self.dight_caps = DigitCaps(
      hparams = self._hparams,
      num_classes = self.num_classes, 
      input_atoms = 8, 
      output_atoms = 16,
      num_routing=self._hparams.routing,
      leaky=self._hparams.leaky,
    )
  
  def call(self, inputs):
    pre_activation = self.conv(inputs)
    primary = self.conv_caps(pre_activation)
    digits = self.dight_caps(primary)




