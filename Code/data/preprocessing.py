import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

DEFAULT_RAW_MNIST_FILENAME = "Code/data/mnist_raw.npz"

def spatial_image(image):
    pass
    # Remember to flip!!!

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

# SOMETHING IS UPSIDE DOWN
def raw_to_spatial_mnist(data, num_images=100, num_x0=28, num_x1=28):
    assert num_x0 * num_x1 == data.shape[1]
    # Input locations:
    X = np.zeros([num_images, num_x0 * num_x1, 2])
    # Convert matrix to grid of input locations:
    x0, x1 = np.linspace(-1, 1, num_x0), np.linspace(-1, 1, num_x1)
    X0, X1 = np.meshgrid(x0, x1)
    X[:] = np.concatenate([X0.reshape(-1, 1), X1.reshape(-1, 1)], axis=1)
    # Brightness values at input locations:
    Y = data[:num_images].reshape([num_images, -1, 1])
    return X, Y

def spatial_image_to_matrices(x, y, num_x0=28, num_x1=28):
    assert len(set([num_x0*num_x1, x.shape[0], y.shape[0]])) == 1
    c = y.reshape(num_x1, num_x0)
    x0 = x[:, 0].reshape(num_x1, num_x0)
    x1 = x[:, 1].reshape(num_x1, num_x0)
    return x0, x1, c


def preprocess_mnist(train_data, train_labels, eval_data, eval_labels):
    # Load training and eval data
    print("Loading")
    print(train_data.shape)
    plt.pcolor(train_data[54].reshape(28,28))
    plt.show()

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
