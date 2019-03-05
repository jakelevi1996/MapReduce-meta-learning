import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import time
from os import mkdir

from models.neuralmodels import BinarySetRegressor, ContinuousSetRegressor
from data.preprocessing import (
    load_raw_mnist, split_binary_image, split_binary_image_batch,
    split_continuous_image, split_continuous_image_batch
)

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

def add_noise(y_condition, noise_prob):
    flip_inds = np.random.binomial(1, noise_prob, y_condition.shape) > 0
    y_noise = np.where(flip_inds, 1.0 - y_condition, y_condition)
    return y_noise, flip_inds

def train_bsr(
    bsr, image_batches, train_ratio=0.9, num_epochs=10, print_every=100,
    plot_every=1000, logdir="results/summaries"
):
    next_print, next_plot = 0, 0

    # Train and save the model
    print("Starting TensorFlow Session...")
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        # Initialise variables
        bsr.initialise_variables(sess)
        # Iterate through each training image:
        t_start, num_images_seen = time(), 0
        for e in range(num_epochs):
            for image_batch in image_batches:
                (
                    x_dark_train, x_light_train, x_test, y_test
                ) = split_binary_image_batch(image_batch, train_ratio)
                loss_val, summary_val = bsr.training_step(
                    sess, x_dark_train, x_light_train, x_test, y_test
                )
                num_images_seen += image_batch.shape[0]
                
                # Add summary to Tensorboard
                if num_images_seen % print_every == 0:
                    writer.add_summary(summary_val, num_images_seen)
                    print("E = {}, NIS = {}, loss = {:.5}.png".format(
                        e, num_images_seen, loss_val
                    ))
                    next_print += print_every
                
                # Plot predictions
                if num_images_seen >= next_plot:
                    # Choose a random image
                    image_ind = np.random.choice(images.shape[0])
                    image = images[image_ind]
                    # Split image
                    x_dark_train, x_light_train, _, _ = split_binary_image(
                        image, train_ratio
                    )
                    # Generate points to evaluate
                    x0, x1 = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
                    X0, X1 = np.meshgrid(x0, x1)
                    mesh_shape = X0.shape
                    X = np.stack([X0.ravel(), X1.ravel()], axis=1)
                    Y = bsr.eval(
                        sess, x_dark_train, x_light_train, X, mesh_shape
                    )
                    # Plot
                    filename = "results/images/Seen {}, TR = {}.png".format(
                        num_images_seen, train_ratio
                    )
                    plot_preds(
                        image, Y, x_dark_train, x_light_train,
                        filename=filename
                    )
                    next_plot += plot_every


    print("\n\nTime taken = {:.5} s".format(time() - t_start))
    print("Train ratio = {}, final loss = {:.5}".format(train_ratio, loss_val))

def train_csr(
    csr, image_batches, train_ratio=0.9, num_epochs=10, print_every=100,
    num_plots=20, model_name="csr", noise_prob=0
):

    # Train and save the model
    print("Starting TensorFlow Session...")
    with tf.Session() as sess:
        logdir = "results/summaries/" + model_name
        writer = tf.summary.FileWriter(logdir, sess.graph)
        # Initialise variables
        csr.initialise_variables(sess)
        # Iterate through each training image:
        t_start, num_images_seen, next_print = time(), 0, 0
        for e in range(num_epochs):
            for image_batch in image_batches:
                (
                    x_condition, y_condition, x_eval, y_eval
                ) = split_continuous_image_batch(image_batch, train_ratio)

                if noise_prob > 0.0:
                    y_noise, _ = add_noise(y_condition, noise_prob)
                else: y_noise = y_condition

                loss_val, summary_val = csr.training_step(
                    sess, x_condition, y_noise, x_eval, y_eval
                )
                num_images_seen += image_batch.shape[0]
                
                # Add summary to Tensorboard
                if num_images_seen >= next_print:
                    writer.add_summary(summary_val, num_images_seen)
                    print("E = {}, NIS = {}, loss = {:.5}.png".format(
                        e, num_images_seen, loss_val
                    ))
                    next_print += print_every

        # If necessary, adjust number of images to plot
        num_plots = min(images.shape[0], num_plots)
        # Choose some random images to plot
        # NB these should really be sampled from test images
        plot_inds = np.random.choice(images.shape[0], num_plots, replace=False)
        for i in plot_inds:
            image = images[i]
            # Split image
            x_condition, y_condition, _, _ = split_continuous_image(
                image, train_ratio
            )
            y_noise, flip_inds = add_noise(y_condition, noise_prob)
            # Generate points and evaluate
            x0, x1 = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
            X0, X1 = np.meshgrid(x0, x1)
            mesh_shape = X0.shape
            X = np.stack([X0.ravel(), X1.ravel()], axis=1)
            Y = csr.eval(
                sess, x_condition, y_noise, X, mesh_shape
            )
            # Plot
            try: mkdir("results/images/{}".format(model_name))
            except FileExistsError: pass
            filename = "results/images/{}/{}.png".format(model_name, time())
            plot_preds(
                image, Y, x_condition=x_condition, flip_inds=flip_inds,
                filename=filename
            )


    print("\n\nTime taken = {:.5} s".format(time() - t_start))
    print("Train ratio = {}, final loss = {:.5}".format(train_ratio, loss_val))
    return loss_val

def sweep_noise_prob():

    # Initialise computational graph
    print("Initialising computational graph...")
    noise_prob_list = np.arange(0.0, 0.30, 0.05)
    loss_list = []

    csr = ContinuousSetRegressor()
    for noise_prob in noise_prob_list:
        model_name = "CSR, noise prob = {:.4}".format(noise_prob)
        loss = train_csr(
            csr, image_batches, noise_prob=noise_prob, model_name=model_name,
            # num_epochs=500
        )
        loss_list.append(loss)
    
    print(noise_prob_list, loss_list)



if __name__ == "__main__":

    print("Loading images...")
    images = load_raw_mnist()[0]
    # images = load_raw_mnist()[0][1:3]
    np.random.shuffle(images)
    batch_size = 100
    split_inds = range(batch_size, images.shape[0], batch_size)
    image_batches = np.array_split(images, split_inds)


    # # Initialise computational graph...
    # print("Initialising computational graph...")
    # # bsr = BinarySetRegressor()
    # csr = ContinuousSetRegressor()
    # # csr = ContinuousSetRegressor(loss_func=tf.losses.absolute_difference)
    # # for train_ratio in np.arange(0.1, 1, 0.1):
    # #     # train_bsr(train_ratio)
    # #     train_csr(csr, image_batches, train_ratio   )
    
    # train_csr(csr, image_batches, num_epochs=500, noise_prob=0.2)
    # # x = 0
    # # y = np.arange(20).reshape(2, 2, 5)
    # # print(add_noise(y, 0.5)[0])

    sweep_noise_prob()
