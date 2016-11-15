from tensorflow.python.ops import rnn, rnn_cell
import tensorflow as tf
from wikipedia_input import wiki_producer


class Config(object):
    vocabulary_size = 10000
    num_steps = 20
    embedding_size = 150
    hidden_size = 650
    num_layers = 2
    keep_prob = 0.5
    batch_size = 30
    learning_rate = 0.0001


class WikiInput(object):
    def __init__(self):
        self.inputs, self.targets = wiki_producer()


class WikiPedia(object):
    def __init__(self, config, wikiinput):
        self.learning_rate = config.learning_rate
        lstm_cell = rnn_cell.BasicLSTMCell(config.hidden_size,
                                           forget_bias=0.,
                                           state_is_tuple=True)
        dropout_cell = rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=config.keep_prob)
        cell = rnn_cell.MultiRNNCell(dropout_cell, state_is_tuple=True)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding",
                                        shape=[config.vocabulary_size, config.embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer(stddev=0.05))
        embed = tf.nn.embedding_lookup(embedding, wikiinput.inputs)
        split = [tf.squeeze(input, squeeze_dims=1) for input in tf.split([1], config.num_steps, embed)]
        state = cell.zero_state(config.batch_size, dtype=tf.float32)
        outputs, _ = rnn.rnn(cell=cell,
                             inputs=split,
                             initial_state=state)
        expand = [tf.expand_dims(output, 1) for output in outputs]
        concat = tf.concat([1], expand)
        softmax_weight = tf.get_variable("softmax_weight",
                                         shape=[config.hidden_size, config.vocabulary_size],
                                         dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.05))
        softmax_bias = tf.get_variable("softmax_weight",
                                       shape=[config.vocabulary_size],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.))
        logits = tf.nn.xw_plus_b(concat, softmax_weight, softmax_bias)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=wikiinput.labels)

    def train(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        return optimizer