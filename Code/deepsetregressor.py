import numpy as np
import tensorflow as tf
from data.preprocessing import load_raw_mnist, raw_to_spatial_mnist
from time import time

def split_image(image, train_ratio=0.9, threshold=0.5, num_x0=28, num_x1=28):
    assert num_x0 * num_x1 == image.size
    assert 0 < train_ratio < 1
    assert 0 < threshold < 1
    # Convert matrix to grid of input locations:
    x0, x1 = np.linspace(-1, 1, num_x0), np.linspace(-1, 1, num_x1)
    X0, X1 = np.meshgrid(x0, x1)
    X = np.concatenate([X0.reshape(-1, 1), X1.reshape(-1, 1)], axis=1)
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

# image = np.arange(15).reshape(3, 5) / 15
# print(image)
# print(split_image(image, num_x0=3, num_x1=5, train_ratio=0.5))



# Define constants
num_hidden_units = 250
learning_rate = 1e-3
num_epochs = 10000
num_images = 300
logdir = "results/summaries"
hidden_activation = tf.nn.relu
# hidden_activation = tf.tanh


print("Loading images...")
images = load_raw_mnist()[0][:num_images]

print(images[0])

"""
# Initialise computational graph...
print("Initialising computational graph...")
# Network for dark pixels
dark_inputs = tf.placeholder(dtype=tf.float32, shape=(None,2))
darknet_hidden_1 = tf.layers.dense(
    inputs=dark_inputs, units=num_hidden_units,
    activation=hidden_activation, name="darknet_hidden_1"
)
darknet_hidden_2 = tf.layers.dense(
    inputs=darknet_hidden_1, units=num_hidden_units,
    activation=hidden_activation, name="darknet_hidden_2"
)
darknet_out = tf.reduce_mean(darknet_hidden_2, axis=0, name="darknet_out")
# Network for light pixels (should call a function shared with dark pixels):
light_inputs = tf.placeholder(dtype=tf.float32, shape=(None,2))
lightnet_hidden_1 = tf.layers.dense(
    inputs=light_inputs, units=num_hidden_units,
    activation=hidden_activation, name="lightnet_hidden_1"
)
lightnet_hidden_2 = tf.layers.dense(
    inputs=lightnet_hidden_1, units=num_hidden_units,
    activation=hidden_activation, name="lightnet_hidden_2"
)
lightnet_out = tf.reduce_mean(lightnet_hidden_2, axis=0, name="lightnet_out")
# Placeholder for test pixels
test_pixel_x = tf.placeholder(dtype=tf.float32, shape=2)
test_pixel_y = tf.placeholder(dtype=tf.float32, shape=[1, 1])
# Concatenate vectors
concat_vector = tf.concat([darknet_out, lightnet_out, test_pixel_x], axis=0)
# Dense layers for deep-set regressor
deepset_hidden = tf.layers.dense(
    inputs=tf.reshape(concat_vector, [1, -1]), units=num_hidden_units,
    activation=hidden_activation, name="deepset_hidden"
)
deepset_logits = tf.layers.dense(
    inputs=deepset_hidden, units=1,
)
deepset_out = tf.sigmoid(deepset_logits, name="deepset_out")


loss_op = tf.losses.sigmoid_cross_entropy(test_pixel_y, deepset_logits)
# Keep it simple; to start with, compute and apply a gradient for each data
# point. Later on accumulate over several batches before applying
optimiser = tf.train.AdamOptimizer(learning_rate)
train_op = optimiser.minimize(loss_op)

init_op = tf.global_variables_initializer()

# Create summaries, for visualising in Tensorboard
summary_op = tf.summary.scalar("Loss", loss_op)

# Machinary for accumulated gradients, starting with a list of trainable vars:
tvs = tf.trainable_variables()
# List of variables initialised to zeros in the shape of each trainable
# variable, which will be used to accumulate gradients:
accum_vars = [tf.Variable(
    # Couldn't tv.initialized_value() be replaced with tv.shape (or similar?)
    initial_value=tf.zeros_like(tv.initialized_value()), trainable=False
) for tv in tvs]
# List of ops for resetting accumulated gradients to zero (again, tf.zeros(tv.shape))
zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
# Op to compute gradient
gvs = optimiser.compute_gradients(loss_op, tvs)
# Op to accumulate gradient
accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
# Op to apply gradient
train_step = optimiser.apply_gradients([
    (accum_vars[i], gv[1]) for i, gv in enumerate(gvs)
])


# Train and save the model
print("Starting TensorFlow Session...")
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)
    # Initialise variables
    sess.run(init_op)
    # Iterate through each training image:
    t_start = time()
    for i, image in enumerate(images):
        sess.run(zero_ops)
        x_dark_train, x_light_train, x_test, y_test = split_image(image)
        # Iterate through pixels:
        # Need to split image up into training and test pixels, and threshold!!
        for x, y in zip(x_test, y_test):
            # Run the graph, summaries and training op
            loss_val, summary_val, _ = sess.run(
                # (loss_op, summary_op, train_op), feed_dict={
                (loss_op, summary_op, accum_ops), feed_dict={
                    dark_inputs: x_dark_train,
                    light_inputs: x_light_train,
                    test_pixel_x: x,
                    test_pixel_y: np.array(y).reshape(1, 1),
                }
            )
            # Add summary to Tensorboard
            writer.add_summary(summary_val, i)
            print(i, loss_val)
        sess.run(train_step)


print("\n\nTime taken = {:.5} s".format(time() - t_start))


TODO:
- Relu
- Accumulate gradients
- batch norm
- gpu
- simpler data set? 
"""