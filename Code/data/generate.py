import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FILENAME = "Code/data/sinusoid_metadataset.npz"
DEFAULT_IMAGE_NAME = "Code/data/sample_data"

def random_x(x_lim=[0, 1], num_tasks=1, num_points=50, xdim=1):
    return np.random.uniform(*x_lim, size=(num_tasks, num_points, xdim))

def deterministic_sinusoid_set(X, phase=0, amplitude=1, freq=1):
    """
    Generate and return an ndarray of sinusoid data with the same shape as
    input data X and the given characteristics.

    phase, amplitude, freq can each be scalars, in which case that set of
    scalars is shared across all tasks, or arrays with shape (num_tasks, 1),
    where num_tasks = X.shape[0], in which case an individual set of phase,
    amplitude and freq is used for each task.
    """
    return amplitude * np.sin(2*np.pi*freq*X + phase)

def random_sinusoid_set(
    num_tasks=10, num_points=50,
    x_lim=[0, 1], phase_lim=[0, 2*np.pi],
    amplitude_lim=[0.5, 2], freq_lim=[0.5, 3],
):
    # Generate random task-specific phase:
    phase = np.random.uniform(*phase_lim, (num_tasks, 1, 1))
    # Generate random task-specific amplitude:
    amplitude = np.random.uniform(*amplitude_lim, (num_tasks, 1, 1))
    # Generate random task-specific frequency:
    freq = np.random.uniform(*freq_lim, (num_tasks, 1, 1))

    X = random_x(x_lim, num_tasks, num_points)
    Y = deterministic_sinusoid_set(X, phase, amplitude, freq)
    
    return X, Y


def generate_sinusoid_metaset_grid():
    pass

def plot_data(x, y, filename=DEFAULT_IMAGE_NAME, format_str='bx'):
    plt.figure()
    plt.plot(x, y, format_str)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # Test generating sinusoids:
    x, y = random_sinusoid_set(num_tasks=1, num_points=20)
    print(x.shape)
    plot_data(x, y)

    # Test generating sinusoid sets:
    num_tasks = 3
    X, Y = random_sinusoid_set(
        num_tasks, num_points=50,
        # phase_min=0, phase_max=1,
        # amplitude_min=1, amplitude_max=1,
        # freq_min=1, freq_max=1,
    )
    print(X.shape)
    print(X[1].shape)
    for i in range(num_tasks):
        x, y = X[i], Y[i]
        print(i, x.shape, y.shape)
        filename = DEFAULT_IMAGE_NAME + "_{}".format(i)
        plot_data(x, y, filename)
