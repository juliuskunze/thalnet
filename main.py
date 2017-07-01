import numpy as np
from sets import Mnist

from gru_baseline import GruBaseline


def plot(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f"Label {label}")
    plt.imshow(image)
    plt.show()


batch_size = 50

baseline = GruBaseline()

train, test = Mnist()

# for image, label in train.sample(batch_size)[:1]:
#    plot(image, label)

baseline.train(get_batch=lambda: train.sample(batch_size))
