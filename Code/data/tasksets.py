import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__": from generate import random_sinusoid_set
else: from data.generate import random_sinusoid_set

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
    
    def plot(self, filename=DEFAULT_IMAGE_NAME):
        plt.figure()
        plt.plot(
            self.train.x, self.train.y, 'bx', self.test.x, self.test.y, 'rx'
        )
        plt.legend(["Training points", "Test points"])
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

class TaskSet():
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def as_list(self):
        # Return arrays in the same order they are taken by `self.__init__`
        return [self.X_train, self.Y_train, self.X_test, self.Y_test]
    
    def get_task(self, task_index):
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

if __name__ == "__main__":
    np.random.seed(4)

    meta_set = SinusoidalMetaSet(num_tasks=5)
    meta_set.save(DEFAULT_FILE_NAME)

    meta_set = MetaSet(DEFAULT_FILE_NAME)

    meta_set.train_tasks.get_task(2).plot()
