from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

from util import define_scope, unzip, single

class MLPClassifier:
    def __init__(self, data, target, dropout,num_hidden=512, num_layers=2):
        self.data = data
        self.target = target
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.prediction
        self.cross_entropy
        self.accuracy
        self.optimize
        self.train_summary
        self.test_summary
        self.weights_summary

    @lazy_property
    def rnn(self):
        lastoutput = self.data
        for _ in range(self.num_layers):
            lastoutput = tf.contrib.layers.fully_connected(lastoutput,num_outputs=self.num_hidden,activation_fn=tf.nn.relu)
        return lastoutput


    @lazy_property
    def logits(self):
        return tf.contrib.layers.fully_connected(self.rnn, num_outputs=int(self.target.shape[1]),
                                                 activation_fn=None)

    @define_scope
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def cross_entropy(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

    @define_scope
    def accuracy(self):
        correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.target, axis=1))
        return tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

    @define_scope
    def optimize(self):
        return tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cross_entropy)

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

    @lazy_property
    def num_parameters(self):
        return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])


def stacked_rnn_cell(num_hidden: int, num_layers=4):
    return tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(num_hidden) for _ in range(num_layers)])


class SequenceClassifier:
    def __init__(self, data, target, dropout,
                 get_rnn_cell: Callable[[], tf.nn.rnn_cell.RNNCell]):
        self.get_rnn_cell = get_rnn_cell
        self.data = tf.reshape(data,shape=[-1,28,28])
        self.target = target
        self.dropout = dropout

        self.prediction
        self.cross_entropy
        self.accuracy
        self.optimize
        self.train_summary
        self.test_summary
        self.weights_summary

    @lazy_property
    def rnn(self):
        #rnn_cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(self.get_rnn_cell(), output_keep_prob=1 - self.dropout)
        #output, last_state = tf.nn.dynamic_rnn(rnn_cell_with_dropout, self.data, dtype=tf.float32)
        output, last_state = tf.nn.dynamic_rnn(self.get_rnn_cell(), self.data, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last_output = tf.gather(output, int(output.shape[0]) - 1)
        return last_output

    @lazy_property
    def logits(self):
        return tf.contrib.layers.fully_connected(self.rnn, num_outputs=int(self.target.shape[1]),
                                                 activation_fn=None)

    @define_scope
    def prediction(self):
        return tf.nn.softmax(self.logits)

    @define_scope
    def cross_entropy(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target))

    @define_scope
    def accuracy(self):
        correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.target, axis=1))
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

    @lazy_property
    def num_parameters(self):
        return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])


class FfGruModule:
    def __init__(self,
                 center_size: int,
                 context_input_size: int,
                 center_output_size: int,
                 input_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 name: str = ''):
        self.name = name

        self.center_size = center_size
        self.context_input_size = context_input_size
        self.center_output_size = center_output_size

        self.input_size = input_size
        self.output_size = output_size

        self.num_gru_units = self.output_size + self.center_output_size

    def __call__(self, input, center_state, module_state):
        """
        :return: output, new_center_features, new_module_state
        """
        with tf.variable_scope(self.name):
            reading_weights = tf.get_variable('reading_weights',shape=[self.center_size,self.context_input_size],initializer=tf.truncated_normal_initializer(stddev=0.1))

            context_input = tf.matmul(center_state, reading_weights)

            module_input = tf.concat([input, context_input], axis=1) if self.input_size else context_input

            inner = tf.contrib.layers.fully_connected(module_input, num_outputs=self.output_size)

            gru = tf.nn.rnn_cell.GRUCell(self.num_gru_units)

            gru_output, new_module_state = gru(inputs=inner, state=module_state)

            output, center_feature_output = tf.split(gru_output,
                                                     [self.output_size, self.center_output_size],
                                                     axis=1) if self.output_size else (None, gru_output)

        return output, center_feature_output, new_module_state


class ThalNetCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 context_input_size: int,
                 center_size_per_module: int,
                 num_modules: int = 4):
        super().__init__(_reuse=None)

        self._context_input_size = context_input_size
        self._input_size = input_size
        self._output_size = output_size
        self._center_size = num_modules * center_size_per_module
        self.center_size_per_module = center_size_per_module
        self._num_modules = num_modules

    @lazy_property
    def state_size(self):
        return [module.center_output_size for module in self.modules] + \
               [module.num_gru_units for module in self.modules]

    @lazy_property
    def output_size(self):
        return self._output_size

    @lazy_property
    def modules(self):
        return [FfGruModule(center_size=self._center_size,
                            context_input_size=self._context_input_size,
                            center_output_size=self.center_size_per_module,
                            input_size=self._input_size if i == 0 else 0,
                            output_size=self.output_size if i == self._num_modules - 1 else 0,
                            name=f'module{i}') for i in range(self._num_modules)]

    def __call__(self, inputs, state, scope=None):
        center_state_per_module = state[:self._num_modules]
        module_states = state[self._num_modules:]

        center_state = tf.concat(center_state_per_module, axis=1)

        outputs, new_center_features, new_module_states = unzip(
            [module(inputs if module.input_size else None, center_state=center_state, module_state=module_state)
             for module, module_state in zip(self.modules, module_states)])

        output = single([o for o in outputs if o is not None])

        return output, new_center_features + new_module_states
