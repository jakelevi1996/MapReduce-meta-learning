import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def plot_preds(
    y_clean, y_condition, condition_inds, y_pred, num_x0_c=28, num_x1_c=28,
    filename="test", verbose=False
):
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(20, 9)
    e = 1 + 1/28
    condition_shape = [num_x0_c, num_x1_c]
    # Plot clean image
    axes[0].imshow(y_clean.reshape(condition_shape), extent=[-e, e, -e, e])
    axes[0].title("Original clean image")
    # Plot noisy conditioning image
    y = np.full(condition_shape, np.nan)
    y.ravel()[condition_inds] = y_condition.ravel()
    cm.get_cmap().set_bad("k")
    axes[1].imshow(y, extent=[-e, e, -e, e])
    axes[1].title("Input noisy conditioning points")
    # Plot predicted image
    axes[2].imshow(y_pred, extent=[-e, e, -e, e])
    axes[2].title("Output model predictions")

    # Save figure
    if verbose: print("Saving figure...")
    plt.tight_layout()
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

def plot_sparse_preds(
    num_x0_c, num_x1_c, y_condition, condition_inds, y_pred,
    filename="filename", verbose=True
):
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)
    # Plot conditioning image
    y = np.full([num_x0_c, num_x1_c], np.nan)
    y.ravel()[condition_inds] = y_condition.ravel()
    cm.get_cmap().set_bad("k")
    axes[0].imshow(y, extent=[-1, 1, -1, 1])
    # Plot predicted image
    axes[1].imshow(y_pred, extent=[-1, 1, -1, 1])
    if verbose: print("Saving figure...")
    plt.savefig(filename)
    plt.close()
