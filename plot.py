import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(title,ys,zs,labels,ylim=(0,1)):
    #Generate a simple plot of the test and training learning curve.

    plt.figure()
    plt.subplot(1,2,1)
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
        plt.plot(y, color=color,
                 label=labels[i])
    plt.subplot(1,2,2)
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Test CrossEntropy")

    for i in range(len(zs)):
        z = zs[i]
        if i >= len(C):
            i -= len(C)
        color = C[i]
        plt.plot(z, color=color,
                 label=labels[i])
       

    plt.legend(loc="best")
    plt.plot()
    plt.show()
    return plt
