import numpy as np
import tensorflow as tf
import keras.layers as L


class AttentionLayer:

    def __init__(self, name, hid_size, activ=tf.tanh):
        """ A layer that computes additive attention response and weights """
        self.name = name
        self.hid_size = hid_size # attention layer hidden units
        self.activ = activ       # attention layer hidden nonlinearity

        with tf.variable_scope(name):
            # create layer variables
            self.linear_e = L.Dense(hid_size)
            self.linear_d = L.Dense(hid_size)
            self.linear_out = L.Dense(1)

    def __call__(self, enc, dec, inp_mask):
        """
        Computes attention response and weights
        :param enc: encoder activation sequence, float32[batch_size, ninp, enc_size]
        :param dec: single decoder state used as "query", float32[batch_size, dec_size]
        :param inp_mask: mask on enc activatons (0 after first eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - attention response vector (weighted sum of enc)
            - probs - attention weights after softmax
        """
        with tf.variable_scope(self.name):

            # Compute logits
            logits_seq = self.linear_out(self.activ(self.linear_e(enc) + \
                                                    self.linear_d(dec)[:, tf.newaxis, :]))
            logits_seq = tf.squeeze(logits_seq, axis = -1)

            # Apply mask - if mask is 0, logits are -inf
            logits_seq = tf.where(inp_mask, logits_seq, tf.fill(tf.shape(logits_seq),
                                                                -np.inf))

            # Compute attention probabilities (softmax)
            probs = tf.nn.softmax(logits_seq)

            # Compute attention response using enc and probs
            attn = tf.reduce_sum(probs[..., tf.newaxis] * enc, axis = 1)

            return attn, probs
