import tensorflow as tf


class WikiPedia(object):
    def __init__(self, inputs):
        vocab_size = inputs.vocab_size
        embedding_size = inputs.embedding_size
        num_units = inputs.num_units
        learning_rate = inputs.learning_rate
        with tf.variable_scope("embedding"):
            embed = tf.get_variable(name="embedding",
                                    shape=[vocab_size, embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    dtype=tf.float32)
            lookup = tf.nn.embedding_lookup(embed, inputs.contexts)
        with tf.variable_scope("rnn"):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm,
                                           inputs=lookup,
                                           sequence_length=inputs.sequence_lengths,
                                           dtype=tf.float32)
            batch_size = outputs.get_shape().as_list()[0]
            reshape = tf.reshape(outputs, [-1, num_units])
        with tf.variable_scope("output"):
            softmax_w = tf.get_variable(name="softmax_w",
                                        shape=[num_units, vocab_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(name="softmax_b",
                                        shape=[vocab_size],
                                        initializer=tf.constant_initializer(value=0.),
                                        dtype=tf.float32)
            xw_plus_b = tf.nn.xw_plus_b(reshape, softmax_w, softmax_b)
            logits = tf.reshape(xw_plus_b, [batch_size, -1, vocab_size])
        with tf.name_scope("loss"):
            fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputs.labels)
            mask = tf.sign(inputs.labels)
            loss_per_example_per_step = tf.mul(fake_loss, mask)
            loss_per_example_sum = tf.reduce_sum(loss_per_example_per_step, reduction_indices=1)
            loss_per_example_average = tf.div(x=loss_per_example_sum,
                                              y=tf.cast(inputs.sequence_lengths, tf.float32))
            self.__loss = tf.reduce_mean(loss_per_example_average, name="loss")
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op


# This class is nearly the same as the former one except that this one calculate the loss with candidate sampling
# because of huge vocab
class Wikipedia(object):
    def __init__(self, inputs):
        vocab_size = inputs.vocab_size
        embedding_size = inputs.embedding_size
        num_units = inputs.num_units
        learning_rate = inputs.learning_rate
        num_sampled = inputs.num_sampled
        num_true = inputs.num_true
        with tf.variable_scope("embedding"):
            embed = tf.get_variable(name="embedding",
                                    shape=[vocab_size, embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    dtype=tf.float32)
            lookup = tf.nn.embedding_lookup(embed, inputs.contexts)
        with tf.variable_scope("rnn"):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm,
                                           inputs=lookup,
                                           sequence_length=inputs.sequence_lengths,
                                           dtype=tf.float32)
            batch_size = outputs.get_shape().as_list()[0]
            reshape = tf.reshape(outputs, [-1, num_units])
        with tf.variable_scope("loss"):
            softmax_w = tf.get_variable(name="softmax_w",
                                        shape=[vocab_size, num_units],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(name="softmax_b",
                                        shape=[vocab_size],
                                        initializer=tf.constant_initializer(value=0.),
                                        dtype=tf.float32)
            nce_loss = tf.nn.nce_loss(weights=softmax_w,
                                      biases=softmax_b,
                                      inputs=reshape,
                                      labels=inputs.labels,
                                      num_classes=vocab_size,
                                      num_sampled=num_sampled,
                                      num_true=num_true)
            mask = tf.sign(inputs.labels)
            fake_loss = tf.mul(tf.reshape(nce_loss, [batch_size, -1]), mask)
            loss_per_example_sum = tf.reduce_sum(fake_loss, reduction_indices=1)
            loss_per_example_average = tf.div(x=loss_per_example_sum,
                                              y=tf.cast(inputs.sequence_lengths))
            self.__loss = tf.reduce_mean(loss_per_example_average, name="loss")
            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op


