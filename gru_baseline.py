import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from sets import Dataset

from tools import timestamp


class GruBaseline:
    def __init__(self,
                 batch_size: int = 50,
                 step_size: int = 28,
                 input_size: int = 28,
                 state_size: int = 10,
                 output_size: int = 10):
        self.step_count = step_size

        step_batch_inputs = [tf.placeholder(tf.float32, (batch_size, input_size), name=f"input{step}") for step in
                             range(step_size)]
        label_batch = tf.placeholder(tf.float32, (batch_size, output_size), name="labels")

        gru = tf.contrib.rnn.GRUCell(state_size)

        initial_state = tf.zeros([batch_size, state_size])
        state = initial_state
        for step_batch in step_batch_inputs:
            # Output is equal to the new state for GRU cells, ignore it:
            _, state = gru(step_batch, state)

        self.logits = tf.contrib.layers.fully_connected(state, output_size, activation_fn=None)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=label_batch))

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)

        correct = tf.equal(tf.arg_max(self.logits, dimension=1), tf.arg_max(label_batch, dimension=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

        self.train_cross_entropy_summary = tf.summary.scalar('cross_entropy_train', self.cross_entropy)
        self.train_accuracy_summary = tf.summary.scalar('accuracy_train', self.accuracy)
        self.train_summary = tf.summary.merge([self.train_cross_entropy_summary, self.train_accuracy_summary])

        self.test_cross_entropy_summary = tf.summary.scalar('cross_entropy_test', self.cross_entropy)
        self.test_accuracy_summary = tf.summary.scalar('accuracy_test', self.accuracy)
        self.test_summary = tf.summary.merge([self.test_cross_entropy_summary, self.test_accuracy_summary])

        self.init = tf.global_variables_initializer()

    def train(self, train: Dataset, test: Dataset, batch_size: int = 50,
              log_path: Path = Path.home() / ".tensorboard" / timestamp()):
        session = tf.Session()

        session.run(self.init)

        summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

        def feed_dict(batch: Dataset) -> dict:
            step_batches = np.swapaxes(batch['data'], 0, 1)

            return dict(
                [(f"input{step}:0", step_batches[step]) for step in range(self.step_count)] +
                [("labels:0", batch['target'])])

        for batch_index in itertools.count():
            _, train_cross_entropy, train_accuracy, train_summary = session.run(
                [self.train_op, self.cross_entropy, self.accuracy, self.train_summary],
                feed_dict(train.sample(batch_size)))

            summary_writer.add_summary(train_summary, batch_index)

            test_cross_entropy, test_accuracy, test_summary = session.run(
                [self.cross_entropy, self.accuracy, self.test_summary], feed_dict(test.sample(batch_size)))

            summary_writer.add_summary(test_summary, batch_index)

            print(
                f"Batch {batch_index}: train accuracy {train_accuracy:.3f} (cross entropy {train_cross_entropy:.3f}), test accuracy {test_accuracy:.3f} (cross entropy {test_cross_entropy:.3f})")
