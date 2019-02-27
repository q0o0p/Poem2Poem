import tensorflow as tf



def infer_length(tok_ids, eos_id):

    # tok_ids: [B, T]

    equal_as_int = lambda x, y: tf.cast(tf.equal(x, y), dtype = tf.int32)

    count_eos = tf.cumsum(equal_as_int(tok_ids, eos_id),
                          axis = 1,
                          exclusive = True) # [B, T]

    return tf.reduce_sum(equal_as_int(count_eos, 0),
                         axis = 1) # [B]
