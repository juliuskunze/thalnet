import numpy as np
import tensorflow as tf

class GruBaseline:
    def __init__(self,
                 batch_size: int = 50,
                 step_size: int = 28,
                 input_size: int = 28,
                 state_size: int = 10,
                 output_size: int = 10):
        self.step_count = step_size

        step_batch_inputs = [tf.placeholder(tf.uint8, (batch_size, input_size), name=f"input{step}") for step in
                             range(step_size)]
        label_batch = tf.placeholder(tf.uint8, (batch_size), name="labels")

        gru = tf.contrib.rnn.GRUCell(state_size)

        initial_state = tf.zeros([batch_size, state_size])
        state = initial_state
        for step_batch in step_batch_inputs:
            state = gru(step_batch, state)

        logits = tf.contrib.layers.fully_connected(state, output_size, activation_fn=None)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(cross_entropy)
        self.init_op = tf.initialize_all_variables()

    def train(self, batches):
        session = tf.Session()

        self.init_op()

        for image_batch, label_batch in batches:
            step_batches = np.swapaxes(image_batch, 0, 1)

            feed_dict = dict(
                [(f"input{step}", step_batches[step]) for step in range(self.step_count)] +
                [("labels", label_batch)])

            session.run(self.train_op, feed_dict)
