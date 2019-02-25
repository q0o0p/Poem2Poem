import math
import tensorflow as tf
import tensorflow.keras.layers as L



class AttentionLayer:

    def __init__(self, hid_size):

        # Create layers
        self._linear_enc = L.Dense(hid_size)
        self._linear_dec = L.Dense(hid_size)
        self._logits = L.Dense(1)


    def __call__(self, enc_seq, dec, inp_mask):

        # enc_seq: [B, T, enc size]
        # dec: [B, dec size]
        # inp_mask: [B, T]

        with tf.name_scope('AttentionLayer'):

            logits_seq = self._logits(tf.tanh(self._linear_enc(enc_seq) +
                                              self._linear_dec(dec)[:, tf.newaxis])) # [B, T, 1]
            logits_seq = tf.squeeze(logits_seq, axis = -1) # [B, T]

            logits_seq = tf.where(inp_mask,
                                  logits_seq,
                                  tf.broadcast_to(-math.inf,
                                                  tf.shape(logits_seq)))

            probs = tf.nn.softmax(logits_seq) # [B, T]

            return tf.reduce_sum(probs[..., tf.newaxis] * enc_seq,
                                 axis = 1) # [B, enc size]
