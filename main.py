from gru_baseline import GruBaseline
from mnist_data import training_batches, plot

batches = training_batches()

baseline = GruBaseline()

baseline.train(batches)

images, labels = batches.__next__()

print(labels)
plot(images[0], labels[0])