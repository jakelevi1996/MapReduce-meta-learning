import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data.preprocessing import load_raw_mnist
from time import time

def split_image(image, train_ratio=0.9, threshold=0.5, num_x0=28, num_x1=28):
    assert num_x0 * num_x1 == image.size
    assert 0 < train_ratio < 1
    assert 0 < threshold < 1
    # Convert matrix to grid of input locations:
    x0, x1 = np.linspace(-1, 1, num_x0), np.linspace(-1, 1, num_x1)
    X0, X1 = np.meshgrid(x0, x1)
    X = np.stack([X0.ravel(), X1.ravel()], axis=1)
    Y = image.reshape([-1, 1])

    # Separate dark and light pixels (careful with Boolean indexing)
    dark_inds = Y > threshold
    y_dark = Y[dark_inds]
    x_dark = X[np.tile(dark_inds, [1, 2])].reshape(-1, 2)
    y_light = Y[np.invert(dark_inds)]
    x_light = X[np.invert(np.tile(dark_inds, [1, 2]))].reshape(-1, 2)
    
    # Separate training and test dark points
    n_dark = y_dark.size
    n_dark_train = int(train_ratio * n_dark)
    dark_train_inds = np.random.choice(n_dark, n_dark_train, replace=False)
    x_dark_train = x_dark[dark_train_inds]
    x_dark_test = np.delete(x_dark, dark_train_inds, axis=0)
    y_dark_test = np.ones(x_dark_test.shape[0], dtype=np.float)

    # Separate training and test light points
    n_light = y_light.size
    n_light_train = int(train_ratio * n_light)
    light_train_inds = np.random.choice(n_light, n_light_train, replace=False)
    x_light_train = x_light[light_train_inds]
    x_light_test = np.delete(x_light, light_train_inds, axis=0)
    y_light_test = np.zeros(x_light_test.shape[0], dtype=np.float)

    # Combine test points
    x_test = np.concatenate([x_light_test, x_dark_test], axis=0)
    y_test = np.concatenate([y_light_test, y_dark_test])
    test_inds = np.arange(y_test.size)
    np.random.shuffle(test_inds)
    x_test = x_test[test_inds]
    y_test = y_test[test_inds]

    return x_dark_train, x_light_train, x_test, y_test

def plot_preds(
    test_image, x_dark_train, x_light_train, Y_pred, filename="test"
):
    # Create figure and axes for subplots
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)
    # Plot input image and conditioning points
    if image.ndim == 1: test_image = test_image.reshape(28, 28)
    axes[0].imshow(test_image, extent=[-1, 1, -1, 1])
    axes[0].plot(x_dark_train[:, 0], -x_dark_train[:, 1], "k+")
    axes[0].plot(x_light_train[:, 0], -x_light_train[:, 1], "w+")
    # Plot network output based on conditioning points
    axes[1].imshow(Y_pred, extent=[-1, 1, -1, 1])
    # Save figure
    # fig.sav
    print("Saving figure...")
    plt.savefig(filename)
    plt.close()


# Define constants
num_hidden_units = 250
learning_rate = 1e-3
num_epochs = 10000
num_images = 10001
print_every = 20
plot_every = 100
train_ratio = 0.9
logdir = "results/summaries"
hidden_activation = tf.nn.relu
# hidden_activation = tf.tanh


print("Loading images...")
images = load_raw_mnist()[0][:num_images]
np.random.shuffle(images)

# Initialise computational graph...
print("Initialising computational graph...")
# Input placeholders
dark_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
light_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
test_pixels_x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
test_pixels_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
num_test_pixels = tf.shape(test_pixels_x)[0]
# Network for dark pixels
darknet_hidden_1 = tf.layers.dense(
    inputs=dark_inputs, units=num_hidden_units,
    activation=hidden_activation, name="darknet_hidden_1"
)
darknet_hidden_2 = tf.layers.dense(
    inputs=darknet_hidden_1, units=num_hidden_units,
    activation=hidden_activation, name="darknet_hidden_2"
)
darknet_out = tf.reduce_mean(
    darknet_hidden_2, axis=0, name="darknet_out", keepdims=True
)
darknet_tiled = tf.tile(
    darknet_out, [num_test_pixels, 1], name="darknet_tiled"
)
# Network for light pixels (should call a function shared with dark pixels):
lightnet_hidden_1 = tf.layers.dense(
    inputs=light_inputs, units=num_hidden_units,
    activation=hidden_activation, name="lightnet_hidden_1"
)
lightnet_hidden_2 = tf.layers.dense(
    inputs=lightnet_hidden_1, units=num_hidden_units,
    activation=hidden_activation, name="lightnet_hidden_2"
)
lightnet_out = tf.reduce_mean(
    lightnet_hidden_2, axis=0, name="lightnet_out", keepdims=True
)
lightnet_tiled = tf.tile(
    lightnet_out, [num_test_pixels, 1], name="lightnet_tiled"
)
# Concatenate vectors
concat_vector = tf.concat(
    [darknet_tiled, lightnet_tiled, test_pixels_x], axis=1,
    name="concat_vector"
)
# Dense layers for deep-set regressor
deepset_hidden = tf.layers.dense(
    inputs=concat_vector, units=num_hidden_units, activation=hidden_activation,
    name="deepset_hidden"
)
deepset_logits = tf.layers.dense(
    inputs=deepset_hidden, units=1, name="deepset_logits"
)
deepset_out = tf.sigmoid(deepset_logits, name="deepset_out")


loss_op = tf.losses.sigmoid_cross_entropy(test_pixels_y, deepset_logits)
# Keep it simple; to start with, compute and apply a gradient for each data
# point. Later on accumulate over several batches before applying
optimiser = tf.train.AdamOptimizer(learning_rate)
train_op = optimiser.minimize(loss_op)

init_op = tf.global_variables_initializer()

# Create summaries, for visualising in Tensorboard
summary_op = tf.summary.scalar("Loss", loss_op)


# Train and save the model
print("Starting TensorFlow Session...")
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)
    # Initialise variables
    sess.run(init_op)
    # Iterate through each training image:
    t_start = time()
    for i, image in enumerate(images):
        x_dark_train, x_light_train, x_test, y_test = split_image(
            image, train_ratio
        )
        # Run the graph, summaries and training op
        loss_val, summary_val, _ = sess.run(
            (loss_op, summary_op, train_op), feed_dict={
                dark_inputs: x_dark_train,
                light_inputs: x_light_train,
                test_pixels_x: x_test,
                test_pixels_y: y_test.reshape(-1, 1),
            }
        )
        # Add summary to Tensorboard
        if i % print_every == 0:
            writer.add_summary(summary_val, i)
            print(i, loss_val)
        
        # Plot predictions
        if i % plot_every == 0:
            x0 = np.linspace(-1, 1, 100)
            x1 = np.linspace(-1, 1, 100)
            X0, X1 = np.meshgrid(x0, x1)
            mesh_shape = X0.shape
            X = np.stack([X0.ravel(), X1.ravel()], axis=1)
            Y = sess.run(deepset_out, feed_dict={
                dark_inputs: x_dark_train,
                light_inputs: x_light_train,
                test_pixels_x: X,
            }).reshape(mesh_shape)
            filename = "results/images/TR = {}, iteration = {}.png".format(
                train_ratio, i
            )
            plot_preds(image, x_dark_train, x_light_train, Y, filename)



print("\n\nTime taken = {:.5} s".format(time() - t_start))
