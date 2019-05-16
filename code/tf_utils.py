import re
import tensorflow as tf


def get_scope_trainable_variables(sub_scope_name = None):
    full_scope_name = tf.get_variable_scope().name
    if full_scope_name != '':
        full_scope_name += '/'
    if sub_scope_name is not None:
        full_scope_name += sub_scope_name + '/'
    return tf.trainable_variables(scope = re.escape(full_scope_name))

def infer_length_exclude_eos(seq, eos_ix, dtype=tf.int32):
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos,axis=1,exclusive=False)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos,0),dtype),axis=1)
    return lengths

# 'infer_length' and 'infer_mask' from 'utils.py':

def infer_length(seq, eos_ix, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_ix), dtype)
    count_eos = tf.cumsum(is_eos,axis=axis,exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos,0),dtype),axis=axis)
    return lengths


def infer_mask(seq, eos_ix, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_ix, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major: mask = tf.transpose(mask)
    return mask
