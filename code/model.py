import re
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

from attention_layer import AttentionLayer



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


        # Create layers

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

        # Attention
        self._attention = AttentionLayer(hid_size = hid_size)


        # Create our architecture in default TF graph
        with tf.name_scope('Seq2SeqModel') as scope_name:

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

        time_steps = tf.shape(self._inp)[1]

        inp_lengths = _infer_length(self._inp, self._inp_eos_id) # [B]
        inp_mask = tf.sequence_mask(inp_lengths,
                                    maxlen = time_steps) # [B, T]

        inp_emb = self._emb_inp(self._inp) # [B, T, emb size]

        with tf.name_scope('Encoder'):
            # enc_last: [B, hid size]
            enc_seq, enc_last = tf.nn.dynamic_rnn(self._enc,
                                                  inp_emb,
                                                  sequence_length = inp_lengths,
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
        bos_one_hot = tf.one_hot(tf.fill([1], self._out_bos_id),
                                 self._out_tok_count) # [1, out toks]
        # First logits will only have '0' (from log(1)) and '-inf' (from log(0)) values:
        first_logits = tf.log(bos_one_hot) # [1, out toks]

        first_logits = tf.broadcast_to(first_logits,
                                       [batch_size, self._out_tok_count]) # [B, out toks]

        target_tok_ids = tf.transpose(target_tok_ids) # [T, B]

        # logits_seq: [T - 1, B, out toks]
        _, logits_seq = tf.scan(lambda accum, elem: self._decode_step(elem, accum[0]),
                                elems = target_tok_ids[:-1],
                                initializer = (self._dec_prev_state, first_logits))

        logits_seq = tf.concat([first_logits[tf.newaxis], logits_seq], axis = 0) # [T, B, out toks]

        return tf.transpose(logits_seq, perm = [1, 0, 2]) # [B, T, out toks]


    def compute_loss(self, target_tok_ids):

        # target_tok_ids: [B, T]

        with tf.name_scope('Seq2SeqModel/loss'):

            time_steps = tf.shape(target_tok_ids)[1]

            target_length = _infer_length(target_tok_ids, self._out_eos_id) # [B]
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
                                     { **dict(zip(self._dec_prev_state, state)),
                                       self._prev_token: out_tok_ids[-1] })
            next_out_tok_id = np.argmax(logits, axis = -1)

            out_tok_ids.append(next_out_tok_id)

            finished |= next_out_tok_id == self._out_eos_id
            if finished.sum() == seq_count:
                break # Early exit if all sequences finished

        out_tok_ids = np.hstack([id[:, np.newaxis] for id in out_tok_ids])

        return out_tok_ids
