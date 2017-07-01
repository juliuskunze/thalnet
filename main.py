import numpy as np
from sets import Mnist

from gru_baseline import GruBaseline


def plot(image: np.ndarray, label: str) -> None:
    from matplotlib import pyplot as plt
    plt.title(f"Label {label}")
    plt.imshow(image)
    plt.show()


# for image, label in train.sample(batch_size)[:1]:
#    plot(image, label)

baseline = GruBaseline()

train, test = Mnist()

baseline.train(train=train, test=test)
