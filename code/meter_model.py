import tensorflow as tf


class MeterModel:

    # -- compute pentameter model loss, given a pentameter input
    def compute_pm_loss(self, is_training, batch_size, enc_hiddens, dec_cell, space_id, pad_id):

        cf             = self.config
        xlen_max       = tf.reduce_max(self.pm_enc_xlen)

        #use decoder hidden states to select encoder hidden states to predict stress for next time step
        repeat_loss    = tf.zeros([batch_size])
        attentions     = tf.zeros([batch_size, xlen_max]) #historical attention weights
        prev_miu       = tf.zeros([batch_size,1])
        outputs        = []
        attention_list = []
        miu_list       = []

        #initial inputs (learnable) and state
        initial_inputs = tf.get_variable("dec_init_input", [cf.pm_enc_dim*2])
        inputs         = tf.reshape(tf.tile(initial_inputs, [batch_size]), [batch_size, -1])
        state          = dec_cell.zero_state(batch_size, tf.float32)

        #manual unroll of time steps because attention depends on previous attention weights
        with tf.variable_scope("RNN"):
            for time_step in range(self.pentameter_len):

                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()

                def attend(enc_hiddens, dec_hidden, attn_hist, prev_miu):
                    with tf.variable_scope("pm_attention"):
                        attend_w = tf.get_variable("attend_w", [cf.pm_enc_dim*2+cf.pm_dec_dim, cf.pm_attend_dim])
                        attend_b = tf.get_variable("attend_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        attend_v = tf.get_variable("attend_v", [cf.pm_attend_dim, 1])
                        miu_w    = tf.get_variable("miu_w", [cf.pm_dec_dim+1, cf.pm_attend_dim])
                        miu_b    = tf.get_variable("miu_b", [cf.pm_attend_dim], initializer=tf.constant_initializer())
                        miu_v    = tf.get_variable("miu_v", [cf.pm_attend_dim, 1])

                    #position attention
                    miu     = tf.minimum(tf.sigmoid(tf.matmul(tf.tanh(tf.matmul(tf.concat(1,
                        [dec_hidden, prev_miu]), miu_w) + miu_b), miu_v)) + prev_miu, tf.ones([batch_size, 1]))
                    miu_p   = miu * tf.reshape(tf.cast(self.pm_enc_xlen-1, tf.float32), [-1, 1])
                    pos     = tf.cast(tf.reshape(tf.tile(tf.range(xlen_max), [batch_size]), [batch_size, -1]),
                        dtype=tf.float32)
                    pos_lp  = -(pos - miu_p)**2 / (2 * tf.reshape(tf.tile([tf.square(cf.sigma)], [batch_size]),
                        [batch_size,-1]))

                    #char encoding attention
                    pos_weight = tf.reshape(tf.exp(pos_lp), [-1, 1])
                    inp_concat = tf.concat(1, [enc_hiddens * pos_weight,
                        tf.reshape(tf.tile(dec_hidden, [1,xlen_max]), [-1,cf.pm_dec_dim])])
                    x       = self.pm_enc_x
                    e       = tf.matmul(tf.tanh(tf.matmul(inp_concat, attend_w) + attend_b), attend_v)
                    e       = tf.reshape(e, [batch_size, xlen_max])
                    mask1   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), pad_id)), dtype=tf.float32)
                    mask2   = tf.cast(tf.equal(x, tf.fill(tf.shape(x), space_id)), dtype=tf.float32)
                    e_mask  = tf.maximum(mask1, mask2)
                    e_mask *= tf.constant(-1e20)

                    #combine alpha with position probability
                    alpha   = tf.nn.softmax(e + e_mask + pos_lp)
                    #alpha   = tf.nn.softmax(e + e_mask)

                    #weighted sum
                    c       = tf.reduce_sum(tf.expand_dims(alpha, 2)*tf.reshape(enc_hiddens,
                        [batch_size, xlen_max, cf.pm_enc_dim*2]), 1)

                    return c, alpha, miu

                dec_hidden, state               = dec_cell(inputs, state)
                enc_hiddens_sum, attn, prev_miu = attend(enc_hiddens, dec_hidden, attentions, prev_miu)

                repeat_loss += tf.reduce_sum(tf.minimum(attentions, attn), 1)
                attentions  += attn
                inputs       = enc_hiddens_sum

                attention_list.append(attn)
                miu_list.append(prev_miu)
                outputs.append(enc_hiddens_sum)

        #reshape output into [batch_size*num_steps,hidden_size]
        outputs = tf.reshape(tf.concat(1, outputs), [-1, cf.pm_enc_dim*2])

        #compute loss
        pm_softmax_w = tf.get_variable("pm_softmax_w", [cf.pm_enc_dim*2, 1])
        pm_softmax_b = tf.get_variable("pm_softmax_b", [1], initializer=tf.constant_initializer())
        pm_logit     = tf.squeeze(tf.matmul(outputs, pm_softmax_w) + pm_softmax_b)
        pm_crossent  = tf.nn.sigmoid_cross_entropy_with_logits(pm_logit,
            tf.tile(tf.cast(self.pentameter, tf.float32), [batch_size]))
        cov_loss     = tf.reduce_sum(tf.nn.relu(self.pm_cov_mask*cf.cov_loss_threshold - attentions), 1)
        pm_cost      = tf.reduce_sum(tf.reshape(pm_crossent, [batch_size, -1]), 1) + \
            cf.repeat_loss_scale*repeat_loss + cf.cov_loss_scale*cov_loss

        #save some variables
        self.pm_logits     = tf.sigmoid(tf.reshape(pm_logit, [batch_size, -1]))
        self.pm_attentions = attention_list
        self.mius          = miu_list

        return pm_cost
