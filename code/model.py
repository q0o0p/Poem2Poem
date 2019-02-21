import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L



def _infer_length(tok_ids, eos_id):

    # tok_ids: [B, T]

    equal_as_int = lambda x, y: tf.cast(tf.equal(x, y), dtype = tf.int32)

    count_eos = tf.cumsum(equal_as_int(tok_ids, eos_id),
                          axis = 1,
                          exclusive = True) # [B, T]

    return tf.reduce_sum(equal_as_int(count_eos, 0),
                         axis = 1) # [B]


class Seq2SeqModel:

    def __init__(self,
                 inp_eos_id, inp_tok_count,
                 out_bos_id, out_eos_id, out_tok_count,
                 emb_size, hid_size):

        # Set fields
        self._inp_eos_id = inp_eos_id
        self._out_bos_id = out_bos_id
        self._out_eos_id = out_eos_id
        self._out_tok_count = out_tok_count

        # Create our architecture in default TF graph
        with tf.name_scope('Seq2SeqModel') as scope_name:

            # Placeholders
            self._inp = tf.placeholder(tf.int32, [None, None]) # [B, T]
            self._prev_token = tf.placeholder(tf.int32, [None]) # [B]

            # Embeddings
            self._emb_inp = L.Embedding(inp_tok_count,
                                        emb_size,
                                        name = 'InputEmbedding')
            self._emb_out = L.Embedding(out_tok_count,
                                        emb_size,
                                        name = 'OutputEmbedding')

            # Encoder
            self._enc = L.GRUCell(hid_size)

            # Decoder
            self._dec = L.GRUCell(hid_size)
            self._dec_logits = L.Dense(self._out_tok_count)
            # Set initial decoder state:
            self._dec_prev_state = self._encode() # [B, hid size]

            # Next state and logits for 'infer' function
            self._next_state_and_logits = self._decode_step() # [B, hid size], [B, out toks]


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

        inp_lengths = _infer_length(self._inp, self._inp_eos_id) # [B]
        inp_emb = self._emb_inp(self._inp) # [B, T, emb size]

        with tf.name_scope('Encoder'):
            # enc_last: [B, hid size]
            _, enc_last = tf.nn.dynamic_rnn(self._enc,
                                            inp_emb,
                                            sequence_length = inp_lengths,
                                            dtype = inp_emb.dtype)

        return enc_last


    def _decode_step(self):

        # self._prev_token: [B]
        # self._dec_prev_state: [B, hid size]

        prev_emb = self._emb_out(self._prev_token[:, tf.newaxis]) # [B, 1, emb size]
        prev_emb = tf.squeeze(prev_emb, axis = 1) # [B, emb size]

        with tf.name_scope('Decoder'):
            # dec_out, dec_state: [B, hid size]
            dec_out, [dec_state] = self._dec(prev_emb, states = [self._dec_prev_state])
            out_logits = self._dec_logits(dec_out) # [B, out toks]

        return dec_state, out_logits


    def _compute_logits(self, target_tok_ids, target_length):

        # target_tok_ids: [B, T]
        # target_length: [B]

        batch_size = tf.shape(target_tok_ids)[0]

        target_emb = self._emb_out(target_tok_ids[:, :-1]) # [B, T-1, emb size]

        # dec_seq: [B, T, hid size]
        dec_seq, _ = tf.nn.dynamic_rnn(self._dec,
                                       target_emb,
                                       sequence_length = target_length - 1,
                                       initial_state = self._dec_prev_state)

        logits_seq = self._dec_logits(dec_seq) # [B, T - 1, out toks]

        # Predict BOS as first logits
        bos_one_hot = tf.one_hot(tf.fill([1], self._out_bos_id),
                                 self._out_tok_count) # [1, out toks]
        # First logits will only have '0' (from log(1)) and '-inf' (from log(0)) values:
        first_logits = tf.log(bos_one_hot) # [1, out toks]

        first_logits = tf.broadcast_to(first_logits[tf.newaxis],
                                       [batch_size, 1, self._out_tok_count]) # [B, 1, out toks]

        return tf.concat((first_logits, logits_seq), axis = 1) # [B, T, out toks]


    def compute_loss(self, target_tok_ids):

        # target_tok_ids: [B, T]

        time_steps = tf.shape(target_tok_ids)[1]

        target_length = _infer_length(target_tok_ids, self._out_eos_id) # [B, T]
        target_mask = tf.sequence_mask(target_length,
                                       maxlen = time_steps) # [B, T]

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
                               fill_value = self._out_bos_id,
                               dtype = np.int32)]
        finished = np.zeros(seq_count, dtype = bool)

        while len(out_tok_ids) - 1 != max_out_tok_count:

            state, logits = sess.run(self._next_state_and_logits,
                                     { self._dec_prev_state: state,
                                       self._prev_token: out_tok_ids[-1] })
            next_out_tok_id = np.argmax(logits, axis = -1)

            out_tok_ids.append(next_out_tok_id)

            finished |= next_out_tok_id == self._out_eos_id
            if finished.sum() == seq_count:
                break # Early exit if all sequences finished

        out_tok_ids = np.hstack([id[:, np.newaxis] for id in out_tok_ids])

        return out_tok_ids
