import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import SequenceClassifier, ThalNetCell, stacked_rnn_cell,MLPClassifier
from util import timestamp
from tensorflow.examples.tutorials.mnist import input_data


def plot(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f'Label {label}')
    plt.imshow(image)
    plt.show()


# for image, label in train.sample(batch_size)[:1]:
#    plot(image, label)

def main(batch_size: int = 100, log_path: Path = Path.home() / 'Desktop/thalnet/tensorboard' / timestamp()):
    mnist = input_data.read_data_sets('~/Desktop/thalnet/mnist/train/',
                                    one_hot=True)
    num_classes = 10
    num_rows, row_size = 28,28
    data = tf.placeholder(tf.float32, [None, num_rows*row_size], name='data')
    target = tf.placeholder(tf.float32, [None, num_classes], name='target')
    dropout = tf.placeholder(tf.float32, name='dropout')

    get_thalnet_cell = lambda: ThalNetCell(input_size=row_size, output_size=num_classes, context_input_size=10,
                                           center_size_per_module=10)

    get_stacked_cell = lambda: stacked_rnn_cell(num_hidden=10,num_layers=4)

    #model = MLPClassifier(data, target, dropout, num_hidden=512)
    #model = SequenceClassifier(data, target, dropout, get_rnn_cell=get_stacked_cell)
    model = SequenceClassifier(data, target, dropout, get_rnn_cell=get_thalnet_cell)

    print(f'{model.num_parameters} parameters')

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

    def get_batch(train,batch_size):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
          xs, ys = mnist.train.next_batch(batch_size)
        else:
          rows = np.random.randint(100,size=batch_size)
          xs, ys = mnist.test.images[:batch_size,:], mnist.test.labels[:batch_size,:]
        return {'x': xs, 'y': ys}

    def feed_dict(batch):
        return {data: batch['x'], target: batch['y'], dropout: 0}

    for batch_index in itertools.count():
        train_batch = get_batch(True,batch_size)

        _, train_cross_entropy, train_accuracy, train_summary = session.run(
            [model.optimize, model.cross_entropy, model.accuracy, model.train_summary],
            feed_dict=feed_dict(train_batch))
        summary_writer.add_summary(train_summary, batch_index)

        if batch_index % 100 == 0:
            test_batch = get_batch(False,batch_size)
            test_cross_entropy, test_accuracy, test_summary = session.run(
            [model.cross_entropy, model.accuracy, model.test_summary], feed_dict=feed_dict(test_batch))

            summary_writer.add_summary(test_summary, batch_index)
            summary_writer.add_summary(session.run(model.weights_summary), batch_index)

        print(
            f'Batch {batch_index}: train accuracy {train_accuracy:.3f} (cross entropy {train_cross_entropy:.3f}), test accuracy {test_accuracy:.3f} (cross entropy {test_cross_entropy:.3f})')
        if batch_index == 1000:
            break

if __name__ == '__main__':
    main()
