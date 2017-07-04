import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from sets import Mnist, Dataset

from baseline import RnnBaseline
from util import timestamp


def plot(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f'Label {label}')
    plt.imshow(image)
    plt.show()


# for image, label in train.sample(batch_size)[:1]:
#    plot(image, label)

def main(batch_size: int = 50, log_path: Path = Path.home() / '.tensorboard' / timestamp()):
    train, test = Mnist()
    _, num_rows, row_size = train.data.shape
    num_classes = train.target.shape[1]
    data = tf.placeholder(tf.float32, [None, num_rows, row_size], name='data')
    target = tf.placeholder(tf.float32, [None, num_classes], name='target')
    dropout = tf.placeholder(tf.float32, name='dropout')

    model = RnnBaseline(data, target, dropout)

    print(f'{model.num_parameters} parameters')

    session = tf.Session()

    session.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(str(log_path), graph=tf.get_default_graph())

    def feed_dict(batch: Dataset) -> dict:
        return {data: batch.data, target: batch.target, dropout: 0}

    for batch_index in itertools.count():
        train_batch = train.sample(batch_size)

        _, train_cross_entropy, train_accuracy, train_summary = session.run(
            [model.optimize, model.cross_entropy, model.accuracy, model.train_summary],
            feed_dict(train_batch))

        summary_writer.add_summary(train_summary, batch_index)

        test_batch = test.sample(batch_size)
        test_cross_entropy, test_accuracy, test_summary = session.run(
            [model.cross_entropy, model.accuracy, model.test_summary], feed_dict(test_batch))

        summary_writer.add_summary(test_summary, batch_index)

        if batch_index % 100 == 0:
            summary_writer.add_summary(session.run(model.weights_summary), batch_index)

        print(
            f'Batch {batch_index}: train accuracy {train_accuracy:.3f} (cross entropy {train_cross_entropy:.3f}), test accuracy {test_accuracy:.3f} (cross entropy {test_cross_entropy:.3f})')


if __name__ == '__main__':
    main()
