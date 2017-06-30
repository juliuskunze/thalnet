from pathlib import Path

import numpy as np
from mnist import MNIST

from tools import paginate

data = MNIST(path=str(Path.home() / "data" / "mnist"))


def training_batches(batch_size: int = 50):
    training_images, training_labels = data.load_training()

    for training_image_batch, training_label_batch in zip(paginate(training_images, batch_size),
                                                          paginate(training_labels, batch_size)):
        yield np.reshape(training_image_batch, (batch_size, 28, 28)), training_label_batch


def plot(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f"Label {label}")
    plt.imshow(image)
    plt.show()
