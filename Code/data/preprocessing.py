import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

DEFAULT_RAW_MNIST_FILENAME = "Code/data/mnist_raw.npz"

def download_raw_mnist(save_path=DEFAULT_RAW_MNIST_FILENAME):
    # NB these methods are deprecated !!!
    logging.info("Loading MNIST...")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    logging.info("Loaded MNIST")
    # Separate arrays
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # Save arrays
    np.savez(save_path, train_data, train_labels, eval_data, eval_labels)

def load_raw_mnist(filename=DEFAULT_RAW_MNIST_FILENAME):
    with np.load(filename) as data:
        train_data, train_labels, eval_data, eval_labels = [
            data[name] for name in data.files
        ]
    
    return train_data, train_labels, eval_data, eval_labels

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
    y_dark_test = np.ones([x_dark_test.shape[0], 1], dtype=np.float)

    # Separate training and test light points
    n_light = y_light.size
    n_light_train = int(train_ratio * n_light)
    light_train_inds = np.random.choice(n_light, n_light_train, replace=False)
    x_light_train = x_light[light_train_inds]
    x_light_test = np.delete(x_light, light_train_inds, axis=0)
    y_light_test = np.zeros([x_light_test.shape[0], 1], dtype=np.float)

    # Combine test points
    x_test = np.concatenate([x_light_test, x_dark_test], axis=0)
    y_test = np.concatenate([y_light_test, y_dark_test], axis=0)
    test_inds = np.arange(y_test.size)
    np.random.shuffle(test_inds)
    x_test = x_test[test_inds]
    y_test = y_test[test_inds]

    return x_dark_train, x_light_train, x_test, y_test

def split_image_batch(
    image_batch, train_ratio=0.9, threshold=0.5, num_x0=28, num_x1=28
):
    x_dark_train_batch, x_light_train_batch = [], []
    x_test_batch, y_test_batch = [], []
    # Split each image separately (could vectorise?)
    for image in image_batch:
        x_dark_train, x_light_train, x_test, y_test = split_image(
            image, train_ratio, threshold, num_x0, num_x1
        )
        x_dark_train_batch.append(x_dark_train)
        x_light_train_batch.append(x_light_train)
        x_test_batch.append(x_test)
        y_test_batch.append(y_test)
    
    # Find minimum number of points in each
    num_dark = min([x_dark.shape[0] for x_dark in x_dark_train_batch])
    num_light = min([x_light.shape[0] for x_light in x_light_train_batch])
    num_test_x = min([x_test.shape[0] for x_test in x_test_batch])
    num_test_y = min([y_test.shape[0] for y_test in y_test_batch])
    assert num_test_x == num_test_y

    # Stack truncated arrays into 3D tensors
    x_dark_train_batch = np.stack([
        x_dark[:num_dark] for x_dark in x_dark_train_batch
    ])
    x_light_train_batch = np.stack([
        x_light[:num_light] for x_light in x_light_train_batch
    ])
    x_test_batch = np.stack([
        x_test[:num_test_x] for x_test in x_test_batch
    ])
    y_test_batch = np.stack([
        y_test[:num_test_y] for y_test in y_test_batch
    ])
    
    return x_dark_train_batch, x_light_train_batch, x_test_batch, y_test_batch


def raw_to_spatial_mnist(data, num_images=1, num_x0=28, num_x1=28):
    assert num_x0 * num_x1 == data.shape[1]
    # Input locations:
    x0, x1 = np.linspace(-1, 1, num_x0), np.linspace(-1, 1, num_x1)
    X0, X1 = np.meshgrid(x0, x1)
    # Are the orders of x and y pixels consistent?
    X = np.stack([X0.ravel(), X1.ravel()], axis=1)
    # Brightness values at input locations:
    if num_images > 1:
        # Should select images using `np.random.choice`?
        Y = data[:num_images].reshape([num_images, -1, 1])
    else: Y = data.reshape([-1, 1])
    return X, Y

def spatial_image_to_matrices(x, y, num_x0=28, num_x1=28):
    assert len(set([num_x0*num_x1, x.shape[0], y.shape[0]])) == 1
    c = y.reshape(num_x1, num_x0)
    x0 = x[:, 0].reshape(num_x1, num_x0)
    x1 = x[:, 1].reshape(num_x1, num_x0)
    return x0, x1, c


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # save_raw_mnist()
    i = 80
    train_data, train_labels, eval_data, eval_labels = load_raw_mnist()
    print(train_data.shape)
    plt.pcolor(np.flip(train_data[i].reshape(28,28), axis=0))
    plt.show()
    X, Y = raw_to_spatial_mnist(train_data)
    plt.pcolor(*spatial_image_to_matrices(X[i], Y[i]))
    plt.show()
