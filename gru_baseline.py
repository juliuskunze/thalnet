import itertools
from pathlib import Path
from typing import Callable

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
        tf.summary.scalar('Cross entropy', self.cross_entropy)

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)

        correct = tf.equal(tf.arg_max(self.logits, dimension=1), tf.arg_max(label_batch, dimension=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
        tf.summary.scalar('Accuracy', self.accuracy)

        self.init_op = tf.global_variables_initializer()

        self.merged_summary_op = tf.summary.merge_all()

    def train(self, get_batch: Callable[[], Dataset], log_path: Path = Path.home() / ".tensorboard" / timestamp()):
        session = tf.Session()

        session.run(self.init_op)

        summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

        for batch_index in itertools.count():
            batch = get_batch()
            step_batches = np.swapaxes(batch['data'], 0, 1)

            feed_dict = dict(
                [(f"input{step}:0", step_batches[step]) for step in range(self.step_count)] +
                [("labels:0", batch['target'])])

            _, cross_entropy, accuracy, summary = session.run(
                [self.train_op, self.cross_entropy, self.accuracy, self.merged_summary_op], feed_dict)

            summary_writer.add_summary(summary, batch_index)

            print(f"Batch {batch_index}: {cross_entropy:.3f} cross entropy, accuracy {accuracy:.3f}")
