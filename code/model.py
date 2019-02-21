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
                 out_tok_count,
                 emb_size, hid_size):

        # Set fields
        self._inp_eos_id = inp_eos_id
        self._out_tok_count = out_tok_count

        # Create our architecture in default TF graph
        with tf.name_scope('Seq2SeqModel'):

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

            # Next state and logits
            self._next_state_and_logits = self._decode_step() # [B, hid size], [B, out toks]


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
