import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

import tf_utils
from attention_layer import AttentionLayer



class TranslationModel:

    def __init__(self,
                 inp_vocab, out_vocab,
                 emb_size, hid_size):

        # Set fields
        self._inp_vocab = inp_vocab
        self._out_vocab = out_vocab


        # Create layers

        # Embeddings
        self._emb_inp = L.Embedding(len(inp_vocab),
                                    emb_size,
                                    name = 'InputEmbedding')
        self._emb_out = L.Embedding(len(out_vocab),
                                    emb_size,
                                    name = 'OutputEmbedding')

        # Encoder
        self._enc = L.GRUCell(hid_size)

        # Decoder
        self._dec = L.GRUCell(hid_size)
        self._dec_logits = L.Dense(len(out_vocab))

        # Attention
        self._attention = AttentionLayer(hid_size = hid_size)


        # Create our architecture in default TF graph
        with tf.name_scope('TranslationModel') as scope_name:

            # Placeholders
            self._inp = tf.placeholder(tf.int32, [None, None]) # [B, T]
            self._prev_token = tf.placeholder(tf.int32, [None]) # [B]

            # Set initial decoder state:
            #  dec_prev_cell_state: [B, hid size],
            #  enc_seq: [B, T, hid size]
            #  inp_mask: [B, T]
            self._dec_prev_state = self._encode()

            # Next state and logits for 'infer' function
            self._next_state_and_logits = self._decode_step(self._prev_token, self._dec_prev_state) # (dec_prev_cell_state, enc_seq, inp_mask), [B, out toks]


            # All trainable variables

            # Need to escape scope name because 'tf.trainable_variables' uses 're.match'
            # and outer scope can contain '.' in its name.
            self._trainable_variables = tf.trainable_variables(re.escape(scope_name))


    @property
    def input(self):
        return self._inp


    @property
    def trainable_variables(self):
        return self._trainable_variables


    def _encode(self):

        # self._inp: [B, T]

        # inp_length: [B]
        # inp_mask: [B, T]
        inp_length, inp_mask = tf_utils.infer_length_and_mask(self._inp,
                                                              self._inp_vocab.eos_id)

        inp_emb = self._emb_inp(self._inp) # [B, T, emb size]

        with tf.name_scope('Encoder'):
            # enc_last: [B, hid size]
            enc_seq, enc_last = tf.nn.dynamic_rnn(self._enc,
                                                  inp_emb,
                                                  sequence_length = inp_length,
                                                  dtype = inp_emb.dtype)

        return enc_last, enc_seq, inp_mask


    def _decode_step(self, prev_token, dec_prev_state):

        # prev_token: [B]
        # dec_prev_state:
        #  dec_prev_cell_state: [B, hid size],
        #  enc_seq: [B, T, hid size]
        #  inp_mask: [B, T]

        dec_prev_cell_state, enc_seq, inp_mask = dec_prev_state

        attn = self._attention(enc_seq, dec_prev_cell_state, inp_mask)

        prev_emb = self._emb_out(prev_token[:, tf.newaxis]) # [B, 1, emb size]
        prev_emb = tf.squeeze(prev_emb, axis = 1) # [B, emb size]

        dec_inputs = tf.concat([prev_emb, attn], axis = 1)

        with tf.name_scope('Decoder'):
            # dec_out, dec_state: [B, hid size]
            dec_out, [dec_state] = self._dec(dec_inputs, states = [dec_prev_cell_state])
            out_logits = self._dec_logits(dec_out) # [B, out toks]

        return (dec_state, enc_seq, inp_mask), out_logits


    def _compute_logits(self, target_tok_ids, target_length):

        # target_tok_ids: [B, T]
        # target_length: [B]

        batch_size = tf.shape(target_tok_ids)[0]

        # Predict BOS as first logits
        bos_one_hot = tf.one_hot(tf.fill([1], self._out_vocab.bos_id),
                                 len(self._out_vocab)) # [1, out toks]
        # First logits will only have '0' (from log(1)) and '-inf' (from log(0)) values:
        first_logits = tf.log(bos_one_hot) # [1, out toks]

        first_logits = tf.broadcast_to(first_logits,
                                       [batch_size, len(self._out_vocab)]) # [B, out toks]

        target_tok_ids = tf.transpose(target_tok_ids) # [T, B]

        # logits_seq: [T - 1, B, out toks]
        _, logits_seq = tf.scan(lambda accum, elem: self._decode_step(elem, accum[0]),
                                elems = target_tok_ids[:-1],
                                initializer = (self._dec_prev_state, first_logits))

        logits_seq = tf.concat([first_logits[tf.newaxis], logits_seq], axis = 0) # [T, B, out toks]

        return tf.transpose(logits_seq, perm = [1, 0, 2]) # [B, T, out toks]


    def compute_loss(self, target_tok_ids):

        # target_tok_ids: [B, T]

        with tf.name_scope('TranslationModel/loss'):

            # target_length: [B]
            # target_mask: [B, T]
            target_length, target_mask = tf_utils.infer_length_and_mask(target_tok_ids,
                                                                        self._out_vocab.eos_id)

            logits_seq = self._compute_logits(target_tok_ids, target_length) # [B, T, out toks]

            logits_seq_masked = tf.boolean_mask(logits_seq, target_mask) # [mask non-zero count, out toks]
            target_tok_ids_masked = tf.boolean_mask(target_tok_ids, target_mask) # [mask non-zero count]

            # loss: [mask non-zero count]
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_tok_ids_masked,
                                                                  logits = logits_seq_masked)

            return tf.reduce_mean(loss)


    def infer(self, inp_tok_ids, max_out_tok_count = None):

        # inp_tok_ids: [B, T]

        sess = tf.get_default_session()

        state = sess.run(self._dec_prev_state, { self._inp: inp_tok_ids })

        seq_count = len(inp_tok_ids)
        out_tok_ids = [np.full(seq_count,
                               fill_value = self._out_vocab.bos_id,
                               dtype = np.int32)]
        finished = np.zeros(seq_count, dtype = bool)

        while len(out_tok_ids) - 1 != max_out_tok_count:

            state, logits = sess.run(self._next_state_and_logits,
                                     { **dict(zip(self._dec_prev_state, state)),
                                       self._prev_token: out_tok_ids[-1] })
            next_out_tok_id = np.argmax(logits, axis = -1)

            out_tok_ids.append(next_out_tok_id)

            finished |= next_out_tok_id == self._out_vocab.eos_id
            if finished.sum() == seq_count:
                break # Early exit if all sequences finished

        out_tok_ids = np.hstack([id[:, np.newaxis] for id in out_tok_ids])

        return out_tok_ids