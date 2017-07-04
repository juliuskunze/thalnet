import numpy as np
import tensorflow as tf
from lazy import lazy

from util import define_scope


class RnnBaseline:
    def __init__(self, data, target, dropout, num_hidden=10, num_layers=4):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout = dropout

        self.data = data
        self.target = target

        self.prediction
        self.cross_entropy
        self.accuracy
        self.optimize
        self.train_summary
        self.test_summary
        self.weights_summary

    @lazy
    def logits(self):
        stacked = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.num_hidden) for _ in range(self.num_layers)])
        stacked_with_dropout = tf.contrib.rnn.DropoutWrapper(stacked, output_keep_prob=1 - self.dropout)
        output, last_state = tf.nn.dynamic_rnn(stacked_with_dropout, self.data, dtype=tf.float32)

        output = tf.transpose(output, [1, 0, 2])
        last_output = tf.gather(output, int(output.shape[0]) - 1)

        return tf.contrib.layers.fully_connected(last_output, num_outputs=int(self.target.shape[1]),
                                                 activation_fn=None)

    @define_scope
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def cross_entropy(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

    @define_scope
    def accuracy(self):
        correct = tf.equal(tf.arg_max(self.logits, dimension=1), tf.arg_max(self.target, dimension=1))
        return tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    @define_scope
    def optimize(self):
        return tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)

    @define_scope('weights')
    def weights_summary(self):
        variables_except_from_optimizer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='^(?!.*optimize).*$')

        return tf.summary.merge([tf.summary.histogram(v.name, v) for v in variables_except_from_optimizer])

    def summary(self):
        test_cross_entropy_summary = tf.summary.scalar(f'cross_entropy', self.cross_entropy)
        test_accuracy_summary = tf.summary.scalar(f'accuracy', self.accuracy)
        return tf.summary.merge([test_cross_entropy_summary, test_accuracy_summary])

    @define_scope('train')
    def train_summary(self):
        return self.summary()

    @define_scope('test')
    def test_summary(self):
        return self.summary()

    @property
    def num_parameters(self):
        return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
