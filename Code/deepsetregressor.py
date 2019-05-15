import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import time
import os

from models.neuralmodels import (
    BinarySetRegressor, ContinuousSetRegressor, ProbabilisticCSR
)
from plotting import (
    plot_preds, plot_results, plot_sparse_preds, plot_probabilistic_preds
)
from data.preprocessing import (
    grid, load_raw_mnist, split_binary_image, split_binary_image_batch,
    split_continuous_image, split_continuous_image_batch,
    gen_sparse_prediction_inputs
)


def add_flip_noise(y_condition, flip_prob):
    flip_inds = np.random.binomial(1, flip_prob, y_condition.shape) > 0
    y_noise = np.where(flip_inds, 1.0 - y_condition, y_condition)
    return y_noise

def add_gaussian_noise(y_condition, noise_std=0.1, low=0.0, high=1.0):
    y_noise = y_condition + np.random.normal(0, noise_std, y_condition.shape)
    y_noise[y_noise < low] = low
    y_noise[y_noise > high] = high
    return y_noise


def train_bsr(
    bsr, train_image_batches, test_images, train_ratio=0.9, num_epochs=10,
    print_every=100, plot_every=1000, logdir="results/summaries"
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
            for image_batch in train_image_batches:
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
                    image_ind = np.random.choice(test_images.shape[0])
                    image = test_images[image_ind]
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

def split_and_add_noise(
    image_batch, train_ratio, max_eval_points, flip_prob, noise_std
):
    # Split image into conditioning set and evaluation set
    x_cnd, y_cnd, x_eval, y_eval = split_continuous_image_batch(
        image_batch, train_ratio, max_eval_points=max_eval_points
    )
    # Add noise to input brightness values
    y_noise = y_cnd
    if flip_prob > 0.0: y_noise = add_flip_noise(y_noise, flip_prob)
    if noise_std > 0.0: y_noise = add_gaussian_noise(y_noise, noise_std)

    return x_cnd, y_noise, x_eval, y_eval

def train_csr(
    csr, train_images, test_images, batch_size=100, train_ratio=0.9,
    num_epochs=10, print_every=1000, num_plots=30, model_name="csr",
    max_eval_points=75, save_model=False, saved_model_path=None, noise_std=0.1,
    flip_prob=0, density_plots=False
):
    # Create directory for tensorboard logging
    logdir = "results/summaries/" + model_name
    while os.path.exists(logdir): logdir += "'"
    os.mkdir(logdir)

    # Train and save the model
    print("Starting TensorFlow Session...")
    loss_val, saved_model_dir = 0.0, None
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        # Restore or initialise variables
        if saved_model_path is not None: csr.restore(sess, saved_model_path)
        else: csr.initialise_variables(sess)
        # Iterate through each training image:
        t_start, num_images_seen, next_print = time(), 0, 0
        for e in range(num_epochs):
            # Shuffle train images and split into batches
            np.random.shuffle(train_images)
            split_inds = range(batch_size, train_images.shape[0], batch_size)
            image_batches = np.array_split(train_images, split_inds)

            # Loop through each batch of images
            for image_batch in image_batches:
                # Split image batch and add noise
                x_cnd, y_noise, x_eval, y_eval = split_and_add_noise(
                    image_batch, train_ratio, max_eval_points, flip_prob,
                    noise_std
                )
                # Perform training step
                csr.training_step(sess, x_cnd, y_noise, x_eval, y_eval)
                num_images_seen += image_batch.shape[0]
                
                # Add summary to Tensorboard and print progress
                if num_images_seen >= next_print:
                    # Choose test images
                    num_test_images = min(batch_size, test_images.shape[0])
                    test_inds = np.random.choice(
                        test_images.shape[0], size=num_test_images,
                        replace=False
                    )
                    test_image_batch = test_images[test_inds]
                    # Split image batch and add noise
                    x_cnd, y_noise, x_eval, y_eval = split_and_add_noise(
                        test_image_batch, train_ratio, max_eval_points,
                        flip_prob, noise_std
                    )
                    # Eval loss and summary, add to tensorboard and print
                    loss_val, summary_val = csr.eval_loss(
                        sess, x_cnd, y_noise, x_eval, y_eval
                    )
                    writer.add_summary(summary_val, num_images_seen)
                    print("E = {}, NIS = {}, loss = {:.5}, t = {:.4} s".format(
                        e, num_images_seen, loss_val, time() - t_start
                    ))
                    next_print += print_every

        if save_model:
            saved_model_dir = "results/saved models/{}".format(model_name)
            while os.path.exists(saved_model_dir): saved_model_dir += "'"
            os.makedirs(saved_model_dir)
            saved_model_path = "{}/{}".format(saved_model_dir, model_name)
            saved_model_path = csr.save(sess, saved_model_path)

        # If necessary, adjust number of images to plot, and folder name
        num_plots = min(test_images.shape[0], num_plots)
        folder_name = "results/images/" + model_name
        while os.path.exists(folder_name): folder_name += "'"
        os.mkdir(folder_name)
        # Choose some random images to plot
        plot_inds = np.random.choice(
            test_images.shape[0], num_plots, replace=False
        )
        for i in plot_inds:
            image = test_images[i]
            # Split image
            x_condition, y_condition, _, _, c_inds = split_continuous_image(
                image, train_ratio, max_eval_points=max_eval_points
            )
            y_noise = add_flip_noise(y_condition, flip_prob)
            y_noise = add_gaussian_noise(y_noise, noise_std)
            # Generate points and evaluate
            num_x0, num_x1 = 100, 100
            X = grid(num_x0, num_x1)
            # Plot
            filename = "{}/{}.png".format(folder_name, time())
            if not density_plots:
                Y = csr.eval_output(sess, x_condition, y_noise, X).reshape(
                    num_x1, num_x0
                )
                plot_preds(
                    image, y_noise, c_inds, Y, filename=filename,
                    verbose=True
                )
            else:
                Y, std = csr.eval_mean_std(
                    sess, x_condition, y_noise, X
                )
                Y, std = Y.reshape(num_x1, num_x0), std.reshape(num_x1, num_x0)
                plot_probabilistic_preds(
                    image, y_noise, c_inds, Y, std, filename=filename,
                    verbose=True
                )


    print("\n\nTime taken = {:.5} s".format(time() - t_start))
    print("Train ratio = {}, final loss = {:.5}".format(train_ratio, loss_val))
    return loss_val, saved_model_path

def sweep_train_ratio(train_images, test_images):
    train_ratio_list = np.arange(0.1, 1, 0.1)
    loss_list = []

    # Initialise computational graph
    print("Initialising computational graph...")
    csr = ContinuousSetRegressor()
    for train_ratio in train_ratio_list:
        model_name = "CSR, TR = {:.4}".format(train_ratio)
        loss, _ = train_csr(
            csr, train_images, test_images, train_ratio=train_ratio,
            model_name=model_name,
        )
        loss_list.append(loss)
    
    print(train_ratio_list, loss_list)
    plot_results(
        train_ratio_list, loss_list, "Train ratio",
        title="Final loss vs train ratio (max 75 eval points)",
        filename="results/sweep_train_ratio"
    )
    np.savetxt(
        "results/sweep_train_ratio.txt",
        [train_ratio_list, loss_list], fmt="%10.5g"
    )
    tf.reset_default_graph()


def sweep_flip_prob(train_images, test_images):
    flip_prob_list = np.arange(0.0, 0.40, 0.05)
    loss_list = []

    print("Initialising computational graph...")
    csr = ContinuousSetRegressor()
    for flip_prob in flip_prob_list:
        model_name = "CSR, flip prob = {:.4}".format(flip_prob)
        loss, _ = train_csr(
            csr, train_images, test_images, flip_prob=flip_prob,
            model_name=model_name,
        )
        loss_list.append(loss)
    
    print(flip_prob_list, loss_list)
    plot_results(
        flip_prob_list, loss_list, "Flip probability",
        title="Final loss vs probability of pixel-flip noise",
        filename="results/sweep_noise_prob"
    )
    np.savetxt(
        "results/sweep_noise_prob.txt",
        [flip_prob_list, loss_list], fmt="%10.5g"
    )
    tf.reset_default_graph()


def sweep_noise_var(train_images, test_images):
    noise_var_list = np.arange(0.0, 0.2, 0.01)
    loss_list = []

    print("Initialising computational graph...")
    csr = ContinuousSetRegressor()
    for noise_std in noise_var_list:
        model_name = "CSR, noise var = {:.4}".format(noise_std)
        loss, _ = train_csr(
            csr, train_images, test_images, noise_std=noise_std,
            model_name=model_name, save_model=True
        )
        loss_list.append(loss)
    
    print(noise_var_list, loss_list)
    plot_results(
        noise_var_list, loss_list, "Noise variance",
        title="Final MSE vs variance of Gaussian noise",
        filename="results/sweep_noise_var"
    )
    np.savetxt(
        "results/sweep_noise_prob.txt",
        [noise_var_list, loss_list], fmt="%10.5g"
    )
    tf.reset_default_graph()


def train_l1(train_images, test_images):
    print("Initialising computational graph...")
    csr = ContinuousSetRegressor(loss_func=tf.losses.absolute_difference)
    model_name = "CSR, L1 loss"
    train_csr(csr, train_images, test_images, model_name=model_name)
    tf.reset_default_graph()

def eval_sparse_predictions(
    saved_model_path, sequential=False, num_cnd_points=5
):
    tf.reset_default_graph()
    csr = ContinuousSetRegressor()
    num_x0_c, num_x1_c = 28, 28
    with tf.Session() as sess:
        csr.restore(sess, saved_model_path)
        x_cnd, y_cnd, c_inds, x_eval = gen_sparse_prediction_inputs(
            num_x0_c, num_x1_c, num_cnd_points
        )
        
        # x_cnd, y_cnd, x_eval, _, c_inds = split_continuous_image(
        #     images[0], 0.9, max_eval_points=5
        # )
        y_cnd_old = y_cnd
        if sequential:
            x_eval = list(x_eval)
            while len(x_eval) > 0:
                x_new = x_eval.pop()
                y_new = csr.eval_output(sess, x_cnd, y_cnd, x_new)
                x_cnd = np.append(x_cnd, [x_new], axis=0)
                y_cnd = np.append(y_cnd, y_new)
            filename="results/predictions (sequential)/{}.png".format(time())
        else:
            filename="results/predictions (parallel)/{}.png".format(time())
        num_x0_e, num_x1_e = 100, 100
        x_eval = grid(num_x0_e, num_x1_e)
        y_pred = csr.eval_output(sess, x_cnd, y_cnd, x_eval)
        y_pred = y_pred.reshape(num_x0_e, num_x1_e)

        plot_sparse_preds(
            num_x0_c, num_x1_c, y_cnd_old, c_inds, y_pred, filename=filename
        )

def eval_recursively_sampled_predictions(
    csr, saved_model_path, num_cnd_points=5, num_images=10
):
    num_x0_c, num_x1_c = 28, 28
    # Restore model
    with tf.Session() as sess:
        csr.restore(sess, saved_model_path)
        # Repeat for each image to be sampled
        for _ in range(num_images):
            x_cnd, y_cnd, c_inds, x_eval = gen_sparse_prediction_inputs(
                num_x0_c, num_x1_c, num_cnd_points
            )
            y_cnd_old = y_cnd
            x_eval = list(x_eval)
            while len(x_eval) > 0:
                x_new = x_eval.pop()
                y_new = csr.sample(sess, x_cnd, y_cnd, x_new)
                x_cnd = np.append(x_cnd, [x_new], axis=0)
                y_cnd = np.append(y_cnd, y_new)
            filename="results/preds (sequentially sampled)/{}.png".format(
                time()
            )
            num_x0_e, num_x1_e = 100, 100
            x_eval = grid(num_x0_e, num_x1_e)
            y_pred = csr.eval_output(sess, x_cnd, y_cnd, x_eval)
            y_pred = y_pred.reshape(num_x0_e, num_x1_e)

            plot_sparse_preds(
                num_x0_c, num_x1_c, y_cnd_old, c_inds, y_pred,
                filename=filename
            )
    
    
    tf.reset_default_graph()

def time_image_batch_split(images, batch_size=100):
    print("Starting `time_image_batch_split` function")
    # Record number of images and start time
    num_images = images.shape[0]
    t_start = time()
    # Shuffle
    np.random.shuffle(images)
    t_shuffle = time() - t_start
    # Split into batches
    split_inds = range(batch_size, train_images.shape[0], batch_size)
    np.array_split(train_images, split_inds)
    t_batch = time() - t_start - t_shuffle
    # Print results
    print("{} images were shuffled in {:.4} s, batched in {:.4} s".format(
        num_images, t_shuffle, t_batch
    ))
    print("Total = {:.4} s".format(t_shuffle + t_batch))
    # Output:
        # 55000 images were shuffled in 0.1721 s, batched in 0.002002 s
        # Total = 0.1741 s



if __name__ == "__main__":

    print("Loading images...")
    train_images, _, test_images, _ = load_raw_mnist()
    # train_images = test_images = train_images[1:3]

    # time_image_batch_split(train_images)
    # np.random.shuffle(train_images)
    # batch_size = 100
    # split_inds = range(batch_size, train_images.shape[0], batch_size)
    # image_batches = np.array_split(train_images, split_inds)
    # saved_model_path="results/saved models/CSR, MSE/CSR, MSE"

    # bsr = BinarySetRegressor()

    # sweep_train_ratio()
    # sweep_noise_prob()
    # sweep_noise_var(train_images, test_images)
    # train_l1()
    # csr = ContinuousSetRegressor()

    csr = ProbabilisticCSR(
        learning_rate=1e-3,
        num_hidden_layers_post_reduce=3,
        num_hidden_layers_pre_reduce=3,
        num_hidden_layers_pre_output=3,
        loss_function="clipped density", var_clip=-1
    )
    # # train_csr(csr, image_batches, model_name="CSR, MSE", save_model=True)
    # loss_val, saved_model_path = train_csr(
    #     csr, train_images, test_images, model_name="full data, var_clip=-1",
    #     save_model=True, print_every=500, num_epochs=10,
    #     # save_model=False, print_every=20, num_epochs=3000,
    #     noise_std=0.1, density_plots=True
    # )

    saved_model_path = "results/saved models/full data, var_clip=-1/full data, var_clip=-1"

    eval_recursively_sampled_predictions(
        csr, saved_model_path, 10
    )

    # for _ in range(10):
    #     eval_sparse_predictions(
    #         saved_model_path, sequential=False, num_cnd_points=50
    #     )
    # train_csr(
    #     csr, image_batches, 0.9, model_name="Gaussian TR 0.9 var 0.1",
    #     num_epochs=0, saved_model_path=saved_model_path, noise_std=0.1
    # )
