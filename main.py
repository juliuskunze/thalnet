import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf

from model import SequenceClassifier, ThalNetCell, GRUCell,MLPClassifier
from util import timestamp
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from plot import plot_learning_curve


def plot_image(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f'Label {label}')
    plt.imshow(image)
    plt.show()

def main(batch_size: int = 100, log_path: Path = Path.home() / 'Desktop/thalnet/tensorboard' / timestamp()):
    mnist = input_data.read_data_sets('mnist',
                                    one_hot=True)
    num_classes = 10
    num_rows, row_size = 28,28

    summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

    def get_batch(train,batch_size):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
          xs, ys = mnist.train.next_batch(batch_size)
        else:
          #rows = np.random.randint(1000,size=batch_size) 
          #xs, ys = mnist.test.images[:rows,:], mnist.test.labels[:rows,:]
          xs, ys = mnist.test.images[:1000,:], mnist.test.labels[:1000,:]
        return {'x': xs, 'y': ys}

    def feed_dict(batch):
        return {data: batch['x'], target: batch['y'], dropout: 0}

    get_thalnet_cell = lambda: ThalNetCell(input_size=row_size, output_size=num_classes, context_input_size=32,
                                           center_size_per_module=32,num_modules=4)

    get_stacked_cell = lambda: GRUCell(num_hidden=50,num_layers=4)

    models = [lambda: MLPClassifier(data, target, dropout, num_hidden=64, num_layers=2),
    lambda: SequenceClassifier(data, target, dropout, get_rnn_cell=get_stacked_cell,num_rows=num_rows, row_size=row_size),
    lambda: SequenceClassifier(data, target, dropout, get_rnn_cell=get_thalnet_cell,num_rows=num_rows, row_size=row_size)
    ]

    
    labels = ['MLP-baseline','GRU-baseline','ThalNet-FF-GRU']
    ys = []
    for model in models:
        with tf.Session() as session:
            data = tf.placeholder(tf.float32, [None, num_rows*row_size], name='data')
            target = tf.placeholder(tf.float32, [None, num_classes], name='target')
            dropout = tf.placeholder(tf.float32, name='dropout')
            model = model()
            # reproduce result under 60,000 total parameters for all three models
            print(f'{model.num_parameters} parameters')
            if model.num_parameters > 60000:
                session.close()
                return
            y = []
            session.run(tf.global_variables_initializer())
            
            for batch_index in itertools.count():
                train_batch = get_batch(True,batch_size)
                _, train_cross_entropy, train_accuracy, train_summary = session.run(
                    [model.optimize, model.cross_entropy, model.accuracy, model.train_summary],
                    feed_dict=feed_dict(train_batch))
                summary_writer.add_summary(train_summary, batch_index)

                if batch_index % 1 == 0:
                    test_batch = get_batch(False,batch_size)
                    test_cross_entropy, test_accuracy, test_summary = session.run(
                    [model.cross_entropy, model.accuracy, model.test_summary], feed_dict=feed_dict(test_batch))

                    summary_writer.add_summary(test_summary, batch_index)
                    summary_writer.add_summary(session.run(model.weights_summary), batch_index)
                y.append(test_accuracy)
                print(
                    f'Batch {batch_index}: train accuracy {train_accuracy:.3f} (cross entropy {train_cross_entropy:.3f}), test accuracy {test_accuracy:.3f} (cross entropy {test_cross_entropy:.3f})')
                if batch_index == 2000:
                    ys.append(y)
                    break
            session.close()
        tf.reset_default_graph()
    plot_learning_curve('Sequential Mnist',ys, labels,ylim=(0,1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
