import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FILENAME = "Code/data/sinusoid_metadataset.npz"

def generate_sinusoid(
    num_tasks=1, num_points=50,
    x_min=0, x_max=1,
    phase=0, amplitude=1, freq=1,
):
    """Generate sinusoid data, and return 2 np.ndarrays with shape
    (num_tasks, num_points) with the given characteristics. X values are
    sampled randomly and uniformly between specified limits.

    phase, amplitude, freq can each be scalars, in which case that set of
    scalars is shared across all tasks, or vectors of length num_task, in which
    case an individual set of phase, amplitude and freq is used for each task
    """
    # Generate matrix of X values
    X = np.random.uniform(x_min, x_max, size=(num_tasks, num_points))
    # Generate matrix of Y values
    # NB X must be transposed, so that task-specific phase, amplitude and freq
    # can be summed automatically along the task axis
    Y = (amplitude * np.sin(2*np.pi*freq*X.T + phase)).T
    return X, Y

def generate_sinusoid_set(
    num_tasks=10, num_points=50,
    x_min=0, x_max=1,
    phase_min=0, phase_max=2*np.pi,
    amplitude_min=0.5, amplitude_max=2,
    freq_min=1, freq_max=2,
):
    phase       = np.random.uniform(phase_min,      phase_max,      num_tasks)
    amplitude   = np.random.uniform(amplitude_min,  amplitude_max,  num_tasks)
    freq        = np.random.uniform(freq_min,       freq_max,       num_tasks)

    X, Y = generate_sinusoid(
        num_tasks, num_points, x_min, x_max, phase, amplitude, freq
    )
    
    return X, Y


def generate_sinusoid_metaset_random(
    filename=DEFAULT_FILENAME,
    num_train_tasks=20, num_train_points=50,
    num_valid_tasks=5, num_valid_points=10,
    num_test_tasks=5, num_test_points=10,
):
    X_metatrain, Y_metatrain = generate_sinusoid_set(
        num_train_tasks, num_train_points
    )
    X_metavalid, Y_metavalid = generate_sinusoid_set(
        num_valid_tasks, num_valid_points
    )
    X_metatest, Y_metatest = generate_sinusoid_set(
        num_test_tasks, num_test_points
    )

    np.savez(
        filename,
        X_metatrain=X_metatrain, Y_metatrain=Y_metatrain,
        X_metavalid=X_metavalid, Y_metavalid=Y_metavalid,
        X_metatest=X_metatest, Y_metatest=Y_metatest,
    )

def generate_sinusoid_metaset_grid():
    pass


if __name__ == "__main__":
    x, y = generate_sinusoid(num_points=20)
    # print(x, y)
    # print(x.shape)
    plt.figure()
    plt.plot(x, y, 'bx')
    plt.grid(True)
    plt.savefig("./Code/data/sample_data")
    plt.close()

    num_tasks = 3
    X, Y = generate_sinusoid_set(
        num_tasks, num_points=50,
        x_min=0, x_max=1,
        # phase_min=0, phase_max=1,
        # amplitude_min=1, amplitude_max=1,
        # freq_min=1, freq_max=1,
    )
    print(X.shape)
    print(X[1].shape)
    for i in range(num_tasks):
        x, y = X[i], Y[i]
        print(i)
        print(x.shape, y.shape)
        plt.figure()
        plt.plot(x, y, 'bx')
        plt.grid(True)
        plt.savefig("./Code/data/sample_data_{}".format(i))
        plt.close()
    generate_sinusoid_metaset_random(
        DEFAULT_FILENAME, 2, 4, 2, 4, 2, 4
    )
    with np.load(DEFAULT_FILENAME) as data:
        print(data.files)
        print(data)
        for f in data.files:
            print(data[f])