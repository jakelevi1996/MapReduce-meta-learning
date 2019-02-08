import numpy as np
import matplotlib.pyplot as plt

# if __name__ == "__main__": from generate import random_sinusoid_set
# else: from data.generate import random_sinusoid_set
# from data.generate import random_sinusoid_set
from generate import random_sinusoid_set
from preprocessing import load_raw_mnist

DEFAULT_IMAGE_NAME = "Code/data/sample_data"
DEFAULT_FILE_NAME = "Code/data/sinusoidal_metataskset.npz"
# ^^ Already defined in `generate.py`

# Generic TaskSet and MetaSet objects:
# (methods and attributes to be inherited by child classes)
class DataSet():
    def __init__(self, x, y): self.x, self.y = x, y

class Task():
    def __init__(self, x_train, y_train, x_test, y_test):
        self.train = DataSet(x_train, y_train)
        self.test = DataSet(x_test, y_test)
    
    def plot(self, filename=DEFAULT_IMAGE_NAME, save_and_close=True):
        plt.figure()
        plt.plot(
            self.train.x, self.train.y, 'bx', self.test.x, self.test.y, 'rx'
        )
        plt.legend(["Training points", "Test points"])
        plt.grid(True)
        if save_and_close:
            plt.savefig(filename)
            plt.close()
    
    def get_x_lims(self):
        x_min = min(*self.train.x, *self.test.x)
        x_max = max(*self.train.x, *self.test.x)
        return x_min, x_max


class TaskSet():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def as_list(self):
        # Return arrays in the same order they are taken by `self.__init__`
        return [self.X_train, self.Y_train, self.X_test, self.Y_test]
    
    def get_num_tasks(self):
        """Implemented as a method so that it is inherited by children, rather
        than including in __init__ which is likely to be overridden"""
        # Create set containing the number of tasks in each task-set array:
        num_tasks_set = set(array.shape[0] for array in [
            self.X_train, self.Y_train, self.X_test, self.Y_test
        ])
        # Check all task-set arrays have the same number of tasks:
        assert len(num_tasks_set) == 1
        # Return that value:
        return num_tasks_set.pop()

    def get_task(self, task_index):
        assert task_index < self.get_num_tasks()
        return Task(
            self.X_train[task_index], self.Y_train[task_index],
            self.X_test[task_index], self.Y_test[task_index]
        )

class MetaSet():
    def __init__(self, filename=None):
        if filename is not None: self.load(filename)
    
    def save(self, filename):
        np.savez(
            filename,
            *self.train_tasks.as_list(),
            *self.valid_tasks.as_list(),
            *self.test_tasks.as_list()
        )
    
    def load(self, filename):
        # TODO: add option to only load a subset of tasks
        with np.load(filename) as data:
            # Load arrays in the same order they are saved by `self.save`
            train_tasks_data = [data[name] for name in data.files[0:4]]
            valid_tasks_data = [data[name] for name in data.files[4:8]]
            test_tasks_data = [data[name] for name in data.files[8:12]]
            self.train_tasks = TaskSet(*train_tasks_data)
            self.valid_tasks = TaskSet(*valid_tasks_data)
            self.test_tasks = TaskSet(*test_tasks_data)

# Data objects specific to sinusoidal data:
class SinusoidalTaskSet(TaskSet):
    """
    Each task consists of training points and test points

    Detailed description coming soon.
    """
    def __init__(
        self, num_train_points=50, num_test_points=30, num_tasks=10,
    ):
        num_points = num_train_points + num_test_points
        # Train and test tasks must be generated jointly such that phase,
        # amplitude and frequency are common to each task:
        X, Y = random_sinusoid_set(num_tasks, num_points)
        self.X_train = X[:, :num_train_points]
        self.Y_train = Y[:, :num_train_points]
        self.X_test = X[:, num_train_points:]
        self.Y_test = Y[:, num_train_points:]

class SinusoidalMetaSet(MetaSet):
    def __init__(self, filename=None, num_tasks=10):
        if filename is not None:
            self.load(filename)
        else:
            self.train_tasks = SinusoidalTaskSet(num_tasks=num_tasks)
            self.valid_tasks = SinusoidalTaskSet(num_tasks=num_tasks)
            self.test_tasks = SinusoidalTaskSet(num_tasks=num_tasks)

class MnistDigit():
    # Should inherit from Task if pixels are partitioned into train and test ?
    
    def __init__(self, y, num_x0=28, num_x1=28):
        """y should be a 2D matrix of brightness values for the image"""
        assert num_x0 * num_x1 == y.size
        # Convert matrix to grid of input locations:
        x0, x1 = np.linspace(-1, 1, num_x0), np.linspace(-1, 1, num_x1)
        X0, X1 = np.meshgrid(x0, x1)
        self.x = np.concatenate([X0.reshape(-1, 1), X1.reshape(-1, 1)], axis=1)
        # Brightness values at input locations:
        self.y = y.reshape([-1, 1])
    
    def as_matrix(self, num_x0=28, num_x1=28):
        # Check matrix dimensions have the right shape
        assert num_x0*num_x1 == self.y.shape[0]
        # Return brightness list as matrix
        return self.y.reshape(num_x1, num_x0)
    
    def plot(
        self, filename=DEFAULT_IMAGE_NAME, save_and_close=True,
        threshold_level=None
    ):
        plt.figure()
        y = np.flipud(self.as_matrix())
        if threshold_level is not None:
            y[y >= threshold_level] = 1.0
            y[y < threshold_level] = 0.0
        plt.pcolormesh(y)
        plt.title("Hello I'm a title")
        plt.axis('off')
        if save_and_close:
            plt.savefig(filename)
            plt.close()


class MnistTaskSet(TaskSet):
    def __init__(self, Y): self.Y = Y
    
    def get_image(self, image_index):
        return MnistDigit(self.Y[image_index])

    def get_num_tasks(self): return self.Y.shape[0]

    # def as_list(self):
    #     # Return arrays in the same order they are taken by `self.__init__`
    #     return [self.X, self.Y]

class MnistMetaSet(MetaSet):
    def __init__(self, filename=None, num_train_tasks=50, num_test_tasks=20):
        if filename is not None:
            # Make sure load and save work!
            self.load(filename)
        else:
            # Load raw data
            train_data, _, eval_data, _ = load_raw_mnist()
            # Create training tasks
            train_inds = np.random.choice(
                train_data.shape[0], size=num_train_tasks, replace=False
            )
            self.train_tasks = MnistTaskSet(train_data[train_inds])
            # Create test tasks
            test_inds = np.random.choice(
                eval_data.shape[0], size=num_test_tasks, replace=False
            )
            self.test_tasks = MnistTaskSet(eval_data[test_inds])

if __name__ == "__main__":
    np.random.seed(3)

    # meta_set = SinusoidalMetaSet(num_tasks=4)
    # meta_set.save(DEFAULT_FILE_NAME)

    # meta_set = MetaSet(DEFAULT_FILE_NAME)

    # meta_set.train_tasks.get_task(3).plot()

    train_data, train_labels, eval_data, eval_labels = load_raw_mnist()
    print(train_data.shape)
    # plt.pcolor(np.flip(train_data[i].reshape(28,28), axis=0))
    print(np.version.version)
    m = MnistDigit(train_data[19])
    m.plot("Code/data/mnist_digit", threshold_level=0.5)

    ms = MnistMetaSet()
    ms.train_tasks.get_image(23).plot("Code/data/mnist_digit2")

