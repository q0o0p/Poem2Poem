import tensorflow as tf
import keras.layers as L

from tf_utils import get_scope_trainable_variables, infer_length
from config_base import ConfigBase
from char_vocab import CharVocab

class RuCharEncoder:
    # This class is needed to create chr representation,
    # which is, like in DeepSpeare,
    # shareable between translator and meter models

    class Config(ConfigBase):
        def __init__(self, **kwargs):
            super().__init__(self,
                             kwargs,
                             emb_size = None,
                             hid_size = None,
                             dropout_prob = None)

    deepspeare_en_config = Config(emb_size = 150,  # as 'char_embedding_dim'
                                  hid_size = 50,   # as 'pm_enc_dim'
                                  #dropout_prob = 1 - 0.7) # as 'keep_prob'
                                  dropout_prob = 0) # to show proper BLEU

    def __init__(self, name, config, is_training):
        # ToDo: deal with 'is_training'

        assert type(name) == str
        assert type(config) == RuCharEncoder.Config

        self.name = name
        self.config = config

        ru_vowels = ['а', 'е', 'ё', 'и', 'о', 'у', 'ы', 'э', 'ю', 'я']
        ru_other_letters = [chr(n) for n in range(ord('а'), ord('я') + 1) if chr(n) not in ru_vowels]
        vocab = CharVocab(chars = ru_vowels + ru_other_letters + ['-', ' ', '@'], reversed_ = False)
        self.space_idx = vocab.token_to_ix[' ']
        self.tok_concat_idx = vocab.token_to_ix['@']
        self._vowel_max_idx = len(ru_vowels) - 1
        self._ru_char_voc = vocab

        with tf.variable_scope(name):

            self._inp = tf.placeholder(tf.int32, [None, None])

            self._emb_inp = L.Embedding(len(self._ru_char_voc), config.emb_size)

            # In original DeepSpeare code they use the same LSTM Cell
            # for both directions
            # Maybe it's better to use two separate cells:
            #self._enc_lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(config.hid_size)
            #self._enc_lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(config.hid_size)

            enc_cell = tf.nn.rnn_cell.LSTMCell(config.hid_size)
            if is_training and config.dropout_prob > 0:
                enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob = 1 - config.dropout_prob)

            self._enc_lstm_fw_cell = enc_cell
            self._enc_lstm_bw_cell = enc_cell

            self._enc_seq, self._en_last_hid = self.encode(self._inp)

            self.trainable_variables = get_scope_trainable_variables()

    def lines_to_char_matrix(self, lines, max_matrix_width):
        return self._ru_char_voc.lines_to_matrix(lines, max_matrix_width)

    def tok_matrix_to_char_matrix(self, tok_voc, tok_matrix, max_matrix_depth):
        return self._ru_char_voc.tok_matrix_to_char_matrix(tok_voc, tok_matrix, max_matrix_depth)

    def vowel_mask(self, inp):
        return inp <= self._vowel_max_idx

    def make_input_feed_dict(self, inp_lines):
        return { self._inp: self._ru_char_voc.lines_to_matrix(inp_lines, max_matrix_width = None) }

    def get_input(self):
        return self._inp

    def get_eos_ix(self):
        return self._ru_char_voc.eos_ix

    def get_encoded_seq(self):
        return self._enc_seq

    def get_encoded_last_hid(self):
        return self._en_last_hid

    def encode(self, inp):
        '''
        Return tuple:
        res[0]: encode seq, float32[batch_size, max_time, cfg.hid_size * 2]
        res[1]: encode last hidden state, float32[batch_size, cfg.hid_size * 2]
        '''
        assert inp.shape.ndims == 2 # [batch_size, max_time]

        inp_lengths = infer_length(inp, self._ru_char_voc.eos_ix)

        inp_emb = self._emb_inp(inp)
        with tf.variable_scope('enc'):
            ((enc_seq_fw,
              enc_seq_bw),
             (enc_last_fw,
              enc_last_bw)) = tf.nn.bidirectional_dynamic_rnn(self._enc_lstm_fw_cell,
                                                              self._enc_lstm_bw_cell,
                                                              inp_emb,
                                                              sequence_length = inp_lengths,
                                                              dtype = inp_emb.dtype)
        enc_seq = tf.concat((enc_seq_fw, enc_seq_bw), axis = -1)
        enc_last_hid = tf.concat((enc_last_fw[1], enc_last_bw[1]), axis = -1)
        return enc_seq, enc_last_hid