import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from sets import Dataset

from tools import timestamp, define_scope


class GruModel:
    def __init__(self,
                 batch_size: int = 50,
                 step_size: int = 28,
                 input_size: int = 28,
                 state_size: int = 10,
                 output_size: int = 10):
        self.batch_size = batch_size
        self.step_size = step_size
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        self.input_batches
        self.label_batch
        self.rnn
        self.logits
        self.prediction
        self.cross_entropy
        self.accuracy
        self.optimize
        self.train_summary
        self.test_summary

    @define_scope
    def input_batches(self):
        return [tf.placeholder(tf.float32, (self.batch_size, self.input_size), name=f'{step}') for step in
                range(self.step_size)]

    @define_scope
    def label_batch(self):
        return tf.placeholder(tf.float32, (self.batch_size, self.output_size), name='0')

    @define_scope
    def rnn(self):
        gru = tf.contrib.rnn.GRUCell(self.state_size)
        initial_state = tf.zeros([self.batch_size, self.state_size])
        state = initial_state

        for step_batch in self.input_batches:
            # Output is equal to the new state for GRU cells, ignore it:
            _, state = gru(step_batch, state)
        return state

    @define_scope
    def logits(self):
        return tf.contrib.layers.fully_connected(self.rnn, self.output_size, activation_fn=None)

    @define_scope
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def accuracy(self):
        correct = tf.equal(tf.arg_max(self.logits, dimension=1), tf.arg_max(self.label_batch, dimension=1))
        return tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    @define_scope
    def cross_entropy(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_batch))

    @define_scope
    def optimize(self):
        return tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)

    @define_scope('train')
    def train_summary(self):
        return self.summary()

    @define_scope('test')
    def test_summary(self):
        return self.summary()

    def summary(self):
        test_cross_entropy_summary = tf.summary.scalar(f'cross_entropy', self.cross_entropy)
        test_accuracy_summary = tf.summary.scalar(f'accuracy', self.accuracy)
        return tf.summary.merge([test_cross_entropy_summary, test_accuracy_summary])

    def train(self, train: Dataset, test: Dataset, batch_size: int = 50,
              log_path: Path = Path.home() / '.tensorboard' / timestamp()):
        session = tf.Session()

        session.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

        def feed_dict(batch: Dataset) -> dict:
            step_batches = np.swapaxes(batch['data'], 0, 1)

            return dict(
                [(f'input_batches/{step}:0', step_batches[step]) for step in range(self.step_size)] +
                [('label_batch/0:0', batch['target'])])

        for batch_index in itertools.count():
            _, train_cross_entropy, train_accuracy, train_summary = session.run(
                [self.optimize, self.cross_entropy, self.accuracy, self.train_summary],
                feed_dict(train.sample(batch_size)))

            summary_writer.add_summary(train_summary, batch_index)

            test_cross_entropy, test_accuracy, test_summary = session.run(
                [self.cross_entropy, self.accuracy, self.test_summary], feed_dict(test.sample(batch_size)))

            summary_writer.add_summary(test_summary, batch_index)

            print(
                f'Batch {batch_index}: train accuracy {train_accuracy:.3f} (cross entropy {train_cross_entropy:.3f}), test accuracy {test_accuracy:.3f} (cross entropy {test_cross_entropy:.3f})')
