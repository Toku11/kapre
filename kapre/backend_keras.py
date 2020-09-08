from tensorflow.keras import backend as K
import tensorflow as tf

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10.0, dtype=x.dtype))
    return numerator / denominator

def amplitude_to_decibel(x, ref_value = 1.0, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).

    Parameters
    ----------
    x: Keras *batch* tensor or variable. It has to be batch because of sample-wise `K.max()`.

    amin: minimum amplitude. amplitude smaller than `amin` is set to this.

    dynamic_range: dynamic_range in decibel

    """
    log_spec = 10.0 * log10(tf.math.maximum(amin, tf.math.square(x)))
    

    if K.ndim(x) > 1:
        axis = tuple(range(K.ndim(x))[1:])
    else:
        axis = None

    log_spec = log_spec - 10.0 * log10(tf.math.maximum(amin,ref_value))
    log_spec = tf.math.maximum(log_spec,
                               tf.math.reduce_max(log_spec,axis=axis,keepdims=True) - dynamic_range)  # [-80, 0]
    return log_spec
    