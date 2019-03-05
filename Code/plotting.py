import matplotlib.pyplot as plt
import numpy as np

def plot_preds(
    test_image, Y_pred, x_dark_train=None, x_light_train=None,
    x_condition=None, flip_inds=None, filename="test", verbose=False
):
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)
    # Plot input image and conditioning points
    if test_image.ndim == 1: test_image = test_image.reshape(28, 28)
    e = 1 + 1/28
    axes[0].imshow(test_image, extent=[-e, e, -e, e])
    if x_dark_train is not None:
        axes[0].plot(x_dark_train[:, 0], -x_dark_train[:, 1], "k+")
    if x_light_train is not None:
        axes[0].plot(x_light_train[:, 0], -x_light_train[:, 1], "w+")
    if x_condition is not None:
        axes[0].plot(x_condition[:, 0], -x_condition[:, 1], "k+")
    if flip_inds is not None:
        axes[0].plot(
            x_condition[:, 0][flip_inds.ravel()],
            -x_condition[:, 1][flip_inds.ravel()], "r+"
        )
    # Plot network output based on conditioning points
    axes[1].imshow(Y_pred, extent=[-e, e, -e, e])
    # Save figure
    if verbose: print("Saving figure...")
    plt.savefig(filename)
    plt.close()

def plot_results(
    x, y, xlabel, ylabel="Final loss", title="title", filename="filename"
):
    plt.figure(figsize=[8, 6])
    plt.plot(x, y, "bo")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.close()