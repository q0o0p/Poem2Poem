import pickle
import numpy as np
import tensorflow as tf
import keras.layers as L
from keras import backend as K

from translation_model_interface import ITranslationModel
from tf_utils import get_scope_trainable_variables, infer_length, infer_mask
from config_base import ConfigBase
from attention_layer import AttentionLayer


class TranslationModel(ITranslationModel):

    class Config(ConfigBase):
        def __init__(self, **kwargs):
            super().__init__(self,
                             kwargs,
                             batch_size = None,
                             emb_size = None,
                             hid_size = None)

    def __init__(self, sess = None, filename = None, name = None, inp_tokenizer = None, out_tokenizer = None, config = None):

        load_from_file = filename is not None
        assert all((p is None) == load_from_file for p in (name, inp_tokenizer, out_tokenizer, config))

        if sess is None:
            sess = tf.Session(graph = tf.Graph())

        self.sess = sess

        with sess.graph.as_default():
            if not load_from_file:
                self.initialize(name, inp_tokenizer, out_tokenizer, config)
            else:
                self.load(filename)


    def initialize(self, name, inp_tokenizer, out_tokenizer, config):

        assert type(config) == TranslationModel.Config

        self.name = name
        self.inp_tokenizer = inp_tokenizer
        self.out_tokenizer = out_tokenizer
        self.inp_voc = inp_tokenizer._vocab
        self.out_voc = out_tokenizer._vocab
        self.config = config
        cfg = config

        with tf.variable_scope(name):

            # define model layers

            self.emb_inp = L.Embedding(len(self.inp_voc), cfg.emb_size)
            self.emb_out = L.Embedding(len(self.out_voc), cfg.emb_size)
            self.enc_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(cfg.hid_size)
            self.enc_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(cfg.hid_size)
            #self.enc0 = tf.nn.rnn_cell.GRUCell(cfg.hid_size)

            self.dec_start = L.Dense(cfg.hid_size)
            self.dec0 = tf.nn.rnn_cell.GRUCell(cfg.hid_size)
            self.dense = L.Dense(cfg.hid_size)
            self.activ = tf.tanh
            self.logits = L.Dense(len(self.out_voc))

            self.attention = AttentionLayer(name = 'attention',
                                            hid_size = 2 * cfg.hid_size)

            # prepare to translate_lines
            self.inp = tf.placeholder('int32', [None, None])
            self.initial_state = self.prev_state = self.encode(self.inp)
            self.prev_tokens = tf.placeholder('int32', [None])
            self.next_state, self.next_logits = self.decode(self.prev_state, self.prev_tokens)
            self.next_softmax = tf.nn.softmax(self.next_logits)

            self.trainable_variables = get_scope_trainable_variables()

        # Call to 'K.get_session()' runs variable initializes for
        # all variables including ones initialized using
        # 'tf.global_variables_initializer()' (at least for Keras
        # 2.0.5) thus it have to be called once here or model weights
        # will be rewritten after training e.g. when 'get_weights' is
        # called.
        K.get_session()

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """

        # encode input sequence, create initial decoder states
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_mask = infer_mask(inp, self.inp_voc.eos_ix, dtype = tf.bool)

        inp_emb = self.emb_inp(inp)
        with tf.variable_scope('enc0'):
            #enc_seq, enc_last = tf.nn.dynamic_rnn(self.enc0,
            #                                      inp_emb,
            #                                      sequence_length = inp_lengths,
            #                                      dtype = inp_emb.dtype)
            ((enc_seq_fw,
              enc_seq_bw),
             ((enc_last_fw_cell_state,
               enc_last_fw_hid_state),
              (enc_last_bw_cell_state,
               enc_last_bw_hid_state))) = tf.nn.bidirectional_dynamic_rnn(self.enc_lstm_fw_cell,
                                                                          self.enc_lstm_bw_cell,
                                                                          inp_emb,
                                                                          sequence_length = inp_lengths,
                                                                          dtype = inp_emb.dtype)
        enc_seq = tf.concat((enc_seq_fw, enc_seq_bw), axis = -1)
        enc_last_states = tf.concat([enc_last_fw_hid_state,
                                     enc_last_bw_hid_state], axis = 1)
        dec_start = self.dec_start(enc_last_states)

        # apply attention layer from initial decoder hidden state
        _, first_attn_probas = self.attention(enc_seq, dec_start, inp_mask)

        # Build first state: include
        # * initial states for decoder recurrent layers
        # * encoder sequence and encoder attn mask (for attention)
        # * make sure that last state item is attention probabilities tensor

        first_state = [dec_start, enc_seq, inp_mask, first_attn_probas]
        return first_state

    def decode(self, prev_state, prev_tokens):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        # Unpack your state: you will get tensors in the same order
        # that you've packed in encode
        [prev_dec, enc_seq, inp_mask, prev_attn_probas] = prev_state


        # Perform decoder step
        # * predict next attn response and attn probas given previous decoder state
        # * use prev token embedding and attn response to update decoder states
        # * (concatenate and feed into decoder cell)
        # * predict logits

        next_attn_response, next_attn_probas = self.attention(enc_seq, prev_dec, inp_mask)

        prev_emb = self.emb_out(prev_tokens[:, tf.newaxis])[:,0]

        dec_inputs = tf.concat([prev_emb, next_attn_response], axis = 1)
        with tf.variable_scope('dec0'):
            new_dec_out, new_dec_state = self.dec0(dec_inputs, prev_dec)
        output_logits = self.logits(self.activ(self.dense(new_dec_out)))
        #output_logits = self.logits(self.activ(new_dec_out))

        # Pack new state:
        # * replace previous decoder state with next one
        # * copy encoder sequence and mask from prev_state
        # * append new attention probas
        next_state = [new_dec_state, enc_seq, inp_mask, next_attn_probas]
        return next_state, output_logits


    def compute_logits(self, inp, out):

        batch_size = tf.shape(inp)[0]

        # Encode inp, get initial state
        first_state = self.encode(inp)

        # initial logits: always predict BOS
        first_logits = tf.log(tf.one_hot(tf.fill([batch_size], self.out_voc.bos_ix),
                                         len(self.out_voc)) + 1e-30)

        # Decode step
        def step(prev_state, y_prev):
            # Given previous state, obtain next state and next token logits
            next_dec_state, next_logits = self.decode(prev_state, y_prev)
            return next_dec_state, next_logits

        # You can now use tf.scan to run step several times.
        # use tf.transpose(out) as elems (to process one time-step at a time)
        # docs: https://www.tensorflow.org/api_docs/python/tf/scan

        out = tf.scan(lambda a, y: step(a[0], y),
                      elems = tf.transpose(out)[:-1],
                      initializer = (first_state, first_logits))


        # FIXME remove?
        #self.sess.run(tf.initialize_all_variables())

        logits_seq = out[1]

        # prepend first_logits to logits_seq
        logits_seq = tf.concat((first_logits[tf.newaxis], logits_seq), axis = 0)

        # Make sure you convert logits_seq from
        # [time, batch, voc_size] to [batch, time, voc_size]
        logits_seq = tf.transpose(logits_seq, perm = [1, 0, 2])

        return logits_seq

    def compute_loss(self, inp, out):

        mask = infer_mask(out, self.out_voc.eos_ix) # [B, T]
        logits_seq = self.compute_logits(inp, out) # [B, T, tokens]

        logits_seq_masked = tf.boolean_mask(logits_seq, mask) # [mask non-zero count, tokens]
        out_masked = tf.boolean_mask(out, mask) # [mask non-zero count]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = out_masked,
                                                                logits = logits_seq_masked)

        return tf.reduce_mean(losses)

    def _make_initial_state_from_matrix(self, inp_matrix):
        return self.sess.run(self.initial_state,
                             {self.inp: inp_matrix})

    def make_initial_state(self, inp_lines):
        return self._make_initial_state_from_matrix(self.inp_tokenizer.lines_to_matrix(inp_lines,
                                                                                       max_matrix_width = None))

    def get_next_state_and_logits(self, state, outputs):

        if type(outputs) == list:
            prev_tokens = np.array([out[-1] for out in outputs], dtype = np.int32)
        else:
            prev_tokens = outputs[:, -1]

        return self.sess.run([self.next_state, self.next_logits],
                             {**dict(zip(self.prev_state, state)),
                              self.prev_tokens: prev_tokens})

    def get_output_tokenizer(self):
        return self.out_tokenizer


    def translate_lines(self, inp_lines, max_output_token_count = None):
        """
        Translates a list of lines by greedily selecting most likely next token at each step
        :returns: a list of output lines, a sequence of model states at each step
        """
        state = self.make_initial_state(inp_lines)
        outputs = self._translate_impl(len(inp_lines), state, max_output_token_count)
        return self.out_tokenizer.matrix_to_lines(outputs)


    def translate_matrix(self, inp_matrix, max_output_token_count = None):
        state = self._make_initial_state_from_matrix(inp_matrix)
        outputs = self._translate_impl(len(inp_matrix), state, max_output_token_count)
        return self.out_tokenizer.matrix_to_lines(outputs)


    def _translate_impl(self, line_count, state, max_output_token_count):
        outputs = [[self.out_voc.bos_ix] for _ in range(line_count)]
        #all_states = [state]
        finished = np.zeros([line_count], bool)

        t = 0
        while True:
            if max_output_token_count is not None and t == max_output_token_count:
                break
            t += 1

            state, logits = self.get_next_state_and_logits(state, outputs)
            next_tokens = np.argmax(logits, axis=-1)
            #all_states.append(state)
            for i in range(len(next_tokens)):
                outputs[i].append(next_tokens[i])
                finished[i] |= next_tokens[i] == self.out_voc.eos_ix

            if finished.sum() == line_count:
                break
        return np.array(outputs) #, all_states

    def dump(self, filename):

        trainable_variable_values = [(var.name, value) for var, value
                                     in zip(self.trainable_variables,
                                            self.sess.run(self.trainable_variables))]

        values = {'name': self.name,
                  'config': self.config.as_dict(),
                  'inp_tokenizer': self.inp_tokenizer,
                  'out_tokenizer': self.out_tokenizer,
                  'trainable_weights': trainable_variable_values}
        pickle.dump(values, open(filename, 'wb'))

    def load(self, filename):
        with open(filename, 'rb') as f:
            values = pickle.load(f)

        trainable_variable_values = values['trainable_weights']
        self.initialize(values['name'], values['inp_tokenizer'], values['out_tokenizer'],
                        TranslationModel.Config(**values['config']))

        assert [var.name for var in self.trainable_variables] == \
               [name for name, value in trainable_variable_values]
        self.sess.run([tf.assign(var, value) for var, (name, value)
                       in zip(self.trainable_variables, trainable_variable_values)])
