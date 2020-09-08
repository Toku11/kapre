import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops

class AdditiveNoise(Layer):
    """
    ### `AdditiveNoise`

    ```python
    kapre.augmentation.AdditiveNoise(power=0.1, random_gain=False, noise_type='white', **kwargs)
    ```
    Add noise to input data and output it.

    #### Parameters
    
    * power: float [scalar]
        - The power of noise. std if it's white noise.
        - Default: ``0.1``

    * random_gain: bool
        - Whether the noise gain is random or not.
        - If ``True``, gain is sampled from ``uniform(low=0.0, high=power)`` in every batch.
        - Default: ``False``

    * noise_type; str,
        - Specify the type of noise. It only supports ``'white'`` now.
        - Default: ```white```


    #### Returns

    Same shape as input data but with additional generated noise.

    """

    def __init__(self, power=0.1, random_gain=False, noise_type='white', **kwargs):
        assert noise_type in ['white']
        self.supports_masking = True
        self.power = power
        self.random_gain = random_gain
        self.noise_type = noise_type
        self.uses_learning_phase = True
        super(AdditiveNoise, self).__init__(**kwargs)
   
        
    def call(self, x, training=None):
        
        if training is None:
            training = K.learning_phase()
            
        if self.random_gain:
            noise_x = x + K.random_normal(
                shape=K.shape(x), 
                mean=0.0, 
                stddev=np.random.uniform(0.0, self.power)
            )
        else:
            noise_x = x + K.random_normal(shape=K.shape(x), 
                                          mean=0.0, 
                                          stddev=self.power)
        output = tf_utils.smart_cond(training,
                                 lambda: array_ops.identity(noise_x),
                                 lambda: array_ops.identity(x))
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'power': self.power,
            'random_gain': self.random_gain,
            'noise_type': self.noise_type,
        }
        base_config = super(AdditiveNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class SpecAugment(Layer):
    """
    ### Spec Augment
    Add masking to input data and output it
    ### Parameters
    x: dB spectrogram
    freq_param: Param of freq masking
    time_param: Param of Time masking
    ## Returns 
    Same shape as input data but with masking in time and frequency 
    """
    def __init__(self, 
                 freq_param = None, 
                 time_param = None,
                 image_data_format:str='default',
                 **kwargs):
        
        self.freq_param = freq_param
        self.time_param = time_param
        assert image_data_format in ('default', 'channels_first', 'channels_last')
        assert (freq_param is not None) or (time_param is not None), "at least one param value should be defined"
        
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format
            
        super(SpecAugment, self).__init__(**kwargs)
        
    def call(self, x, training=None):
        
        if training is None:
            training = K.learning_phase()
            
        if self.image_data_format == 'channels_first':
            assert x.shape[1] == 1, 'SpecAugment does not support 2D images yet'

        else:
            assert x.shape[3] == 1, 'SpecAugment does not support 2D images yet'
        
        if self.freq_param:
            x = tf_utils.smart_cond(training, 
                                     lambda: self.random_masking_along_axis(x, 
                                                                            param=self.freq_param, 
                                                                            axis = 0),
                                     lambda: array_ops.identity(x))
        if self.time_param:
            x = tf_utils.smart_cond(training, 
                                     lambda: self.random_masking_along_axis(x, 
                                                                            param=self.time_param,
                                                                            axis = 1),
                                     lambda: array_ops.identity(x))
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            'freq_param': self.freq_param,
            'time_param': self.time_param,
            'image_data_format': self.image_data_format,
        }
        base_config = super(SpecAugment, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def random_masking_along_axis(self, input_f, param, axis, name='masking'):
        """
        Apply masking to a spectrogram in the time/freq domain.
        Args:
          input: An audio spectogram.
          param: Parameter of freq masking.
          name: A name for the operation (optional).
        Returns:
          A tensor of spectrogram.
        """
        # TODO: Support audio with channel > 1.
        _max = tf.shape(input_f)[axis + 1]
        _shape = [-1, 1, 1, 1]
        _shape[axis + 1] = _max
        f = tf.random.uniform(shape=(), minval=0, maxval=param, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(
            shape=(), minval=0, maxval=_max - f, dtype=tf.dtypes.int32
        )
        indices = tf.reshape(tf.range(_max), tuple(_shape))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
        )
        return tf.compat.v2.where(condition, 0.0, input_f)
    