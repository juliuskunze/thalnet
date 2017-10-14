import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(title,ys, labels,ylim=(0,1)):
    #Generate a simple plot of the test and training learning curve.

    plt.figure()
    plt.title(title)
    C = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Steps")
    plt.ylabel("Test Accuracy")

    for i in range(len(ys)):
        y = ys[i]
        if i >= len(C):
            i -= len(C)
        color = C[i]
        y_mean = np.mean(y)
        y_std = np.std(y)
        plt.fill_between(x=list(range(len(y))),y1=y_mean - y_std,
                         y2=y_mean + y_std, alpha=0.1,
                         color=color)
        plt.plot(y_mean, color=color,
                 label=labels[i])

    plt.legend(loc="best")
    plt.plot()
    plt.show()
    return plt
