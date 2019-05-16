import numpy as np
import tensorflow as tf

from tf_utils import get_scope_trainable_variables, infer_length_exclude_eos, infer_length, infer_mask
from config_base import ConfigBase


class MeterModel:
    # Neural architecture of meter model
    # and most of code
    # are borrowed from DeepSpeare Pentameter model

    class Config(ConfigBase):
        def __init__(self, **kwargs):
            super().__init__(self,
                             kwargs,
                             max_out_length = None,
                             # From deepspeare code:
                             dropout_prob = None,
                             pm_dec_dim = None,
                             max_grad_norm = None,
                             pm_learning_rate = None,
                             pm_attend_dim = None,
                             sigma = None,
                             cov_loss_threshold = None,
                             repeat_loss_scale = None,
                             cov_loss_scale = None)

    deepspeare_config = Config(max_out_length = 15,
                               # From deepspeare code:
                               #dropout_prob = 1 - 0.7, # as 'keep_prob'
                               dropout_prob = 0, # to show proper BLEU
                               pm_dec_dim=200,
                               pm_attend_dim=50,
                               pm_learning_rate=0.001,
                               repeat_loss_scale=1.0,
                               cov_loss_scale=1.0,
                               cov_loss_threshold=0.7,
                               sigma=1.00,
                               max_grad_norm=5)

    def __init__(self, name, config, batch_size, is_training, ru_char_encoder):

        assert type(name) == str
        assert type(config) == MeterModel.Config

        self.name = name
        self.config = config

        self._ru_char_encoder = ru_char_encoder
        self._stress_eos_ix = -1

        cfg = self.config
        cfg_pm_enc_dim = ru_char_encoder.config.hid_size

        with tf.variable_scope(name):

            self._out = tf.placeholder(tf.int32, [None, None])

            dec_cell = tf.nn.rnn_cell.LSTMCell(cfg.pm_dec_dim)
            #if is_training and cfg.dropout_prob > 0:
            #    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob = 1 - cfg.dropout_prob)

            enc_hiddens = ru_char_encoder.get_encoded_seq()
            enc_hiddens  = tf.reshape(enc_hiddens, [-1, cfg_pm_enc_dim*2]) #[batch_size*num_steps, hidden]

            #if not is_training:
            self.pm_costs     = self.compute_pm_loss(batch_size = batch_size,
                                                     enc_hiddens = enc_hiddens,
                                                     dec_cell = dec_cell)
            self.pm_mean_cost = tf.reduce_sum(self.pm_costs) / batch_size

            self.trainable_variables = get_scope_trainable_variables() + \
                                       ru_char_encoder.trainable_variables

        if is_training:
            #run optimiser and backpropagate (clipped) gradients for pm loss
            pm_tvars         = self.trainable_variables
            pm_grads, _      = tf.clip_by_global_norm(tf.gradients(self.pm_mean_cost, pm_tvars),
                                                      cfg.max_grad_norm)
            self.pm_train_op = tf.train.AdamOptimizer(cfg.pm_learning_rate).apply_gradients(zip(pm_grads, pm_tvars))

    def get_input(self):
        return self._ru_char_encoder.get_input()

    def get_output(self):
        return self._out

    def to_stress_matrix(self, stress_lines):

        max_len = min(self.config.max_out_length, max(map(len, stress_lines)))

        matrix = np.full((len(stress_lines), max_len), fill_value = self._stress_eos_ix, dtype = np.int32)
        for i, stresses in enumerate(stress_lines):
            stresses = stresses[:max_len]
            matrix[i, :len(stresses)] = stresses

        return matrix

    # -- compute pentameter model loss, given a pentameter input
    # It may seem strange that we pass dec_cell here despite it is
    def compute_pm_loss(self, batch_size, enc_hiddens, dec_cell):

        # Note: Deepspeare uses old TensorFlow API where tf.concat accepts axis first like: tf.concat(1, [t1, t2])
        # So, here it is changed to new TF style: tf.concat([t1, t2], axis = 1)

        cfg = self.config
        cfg_pm_enc_dim = self._ru_char_encoder.config.hid_size

        space_id = self._ru_char_encoder.space_idx

        inp = self.get_input()
        out = self.get_output()
        out_len = infer_length_exclude_eos(out, self._stress_eos_ix)
        max_out_len = tf.shape(out)[1]
        out_mask = tf.sequence_mask(out_len, maxlen = max_out_len, dtype = tf.float32)

        eos_ix = self._ru_char_encoder.get_eos_ix()
        inp_lengths = infer_length(inp, eos_ix)
        inp_mask = infer_mask(inp, eos_ix, dtype = tf.bool)
        pm_cov_mask = tf.cast(self._ru_char_encoder.vowel_mask(inp) & inp_mask, dtype = tf.float32)
        #xlen_max       = tf.reduce_max(inp_lengths) # Logic in Deepspeare
        xlen_max       = tf.shape(inp)[1] # Our current logic

        #use decoder hidden states to select encoder hidden states to predict stress for next time step
        repeat_loss    = tf.zeros([batch_size])
        attentions     = tf.zeros([batch_size, xlen_max]) #historical attention weights
        prev_miu       = tf.zeros([batch_size,1])
        outputs        = []
        attention_list = []
        miu_list       = []

        #initial inputs (learnable) and state
        initial_inputs = tf.get_variable("dec_init_input", [cfg_pm_enc_dim*2])
        inputs         = tf.reshape(tf.tile(initial_inputs, [batch_size]), [batch_size, -1])
        state          = dec_cell.zero_state(batch_size, tf.float32)

        #manual unroll of time steps because attention depends on previous attention weights
        with tf.variable_scope("RNN"):
            for time_step in range(cfg.max_out_length):

                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                def attend(enc_hiddens, dec_hidden, attn_hist, prev_miu):
                    with tf.variable_scope("pm_attention"):
                        attend_w = tf.get_variable("attend_w", [cfg_pm_enc_dim*2+cfg.pm_dec_dim, cfg.pm_attend_dim])
                        attend_b = tf.get_variable("attend_b", [cfg.pm_attend_dim], initializer=tf.constant_initializer())
                        attend_v = tf.get_variable("attend_v", [cfg.pm_attend_dim, 1])
                        miu_w    = tf.get_variable("miu_w", [cfg.pm_dec_dim+1, cfg.pm_attend_dim])
                        miu_b    = tf.get_variable("miu_b", [cfg.pm_attend_dim], initializer=tf.constant_initializer())
                        miu_v    = tf.get_variable("miu_v", [cfg.pm_attend_dim, 1])

                    #position attention
                    miu     = tf.minimum(tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(tf.concat(
                        [dec_hidden, prev_miu], axis = 1), miu_w) + miu_b), miu_v)) + prev_miu, tf.ones([batch_size, 1]))
                    miu_p   = miu * tf.reshape(tf.cast(inp_lengths-1, tf.float32), [-1, 1])
                    pos     = tf.cast(tf.reshape(tf.tile(tf.range(xlen_max), [batch_size]), [batch_size, -1]),
                        dtype=tf.float32)
                    pos_lp  = -(pos - miu_p)**2 / (2 * tf.reshape(tf.tile([tf.square(cfg.sigma)], [batch_size]),
                        [batch_size,-1]))

                    #char encoding attention
                    pos_weight = tf.reshape(tf.exp(pos_lp), [-1, 1])
                    inp_concat = tf.concat([enc_hiddens * pos_weight,
                        tf.reshape(tf.tile(dec_hidden, [1,xlen_max]), [-1,cfg.pm_dec_dim])], axis = 1)
                    x       = inp
                    e       = tf.matmul(tf.tanh(tf.matmul(inp_concat, attend_w) + attend_b), attend_v)
                    e       = tf.reshape(e, [batch_size, xlen_max])
                    mask1   = tf.cast(~inp_mask, dtype=tf.float32)
                    mask2   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), space_id)), dtype=tf.float32)
                    e_mask  = tf.maximum(mask1, mask2)
                    e_mask *= tf.constant(-1e20)

                    #combine alpha with position probability
                    alpha   = tf.nn.softmax(e + e_mask + pos_lp)
                    #alpha   = tf.nn.softmax(e + e_mask)

                    #weighted sum
                    c       = tf.reduce_sum(tf.expand_dims(alpha, 2)*tf.reshape(enc_hiddens,
                        [batch_size, xlen_max, cfg_pm_enc_dim*2]), 1)

                    return c, alpha, miu

                dec_hidden, state               = dec_cell(inputs, state)
                enc_hiddens_sum, attn, prev_miu = attend(enc_hiddens, dec_hidden, attentions, prev_miu)

                # Zero 'attn' if past end of output:
                valid_step = time_step < out_len
                attn *= tf.cast(valid_step, tf.float32)[:, tf.newaxis]

                repeat_loss += tf.reduce_sum(tf.minimum(attentions, attn), 1)
                attentions  += attn
                inputs       = enc_hiddens_sum

                attention_list.append(attn)
                miu_list.append(prev_miu)
                outputs.append(enc_hiddens_sum)

        #reshape output into [batch_size*num_steps,hidden_size]
        #outputs = tf.reshape(tf.concat(outputs, axis = 1), [-1, cfg_pm_enc_dim*2]) # Original code
        outputs = tf.concat(outputs, axis = 1)
        outputs = outputs[:, :max_out_len * cfg_pm_enc_dim*2] # Also truncate outputs
        outputs = tf.reshape(outputs, [-1, cfg_pm_enc_dim*2])


        #compute loss
        pm_softmax_w = tf.get_variable("pm_softmax_w", [cfg_pm_enc_dim*2, 1])
        pm_softmax_b = tf.get_variable("pm_softmax_b", [1], initializer=tf.constant_initializer())
        #pm_logit     = tf.squeeze(tf.matmul(outputs, pm_softmax_w) + pm_softmax_b) # Original code
        pm_logit     = tf.reshape(tf.matmul(outputs, pm_softmax_w) + pm_softmax_b, [batch_size, -1])
        pm_crossent  = tf.nn.sigmoid_cross_entropy_with_logits(logits = pm_logit,
            #labels = tf.tile(tf.cast(fixed_out, tf.float32), [batch_size])) # Original code
            labels = tf.cast(out, tf.float32))
        pm_crossent *= out_mask # Addition: Mask out past-end output
        cov_loss     = tf.reduce_sum(tf.nn.relu(pm_cov_mask*cfg.cov_loss_threshold - attentions), 1)
        #pm_cost      = tf.reduce_sum(tf.reshape(pm_crossent, [batch_size, -1]), 1) + \ # Original code
        # No need to reshape 'pm_crossent' now:
        pm_cost      = tf.reduce_sum(pm_crossent, 1) + \
            cfg.repeat_loss_scale*repeat_loss + cfg.cov_loss_scale*cov_loss

        #save some variables
        self.pm_logits     = tf.sigmoid(tf.reshape(pm_logit, [batch_size, -1]))
        self.pm_attentions = attention_list
        self.mius          = miu_list

        return pm_cost
