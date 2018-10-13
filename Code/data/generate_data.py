import numpy as np
import matplotlib.pyplot as plt

def generate_sinusoid(
    num_points=50, phase=0, amplitude=1, freq=1, x_min=0, x_max=1
):
    """Generate sinusoid data, and return 2 np.ndarrays with shape (num_points)
    with the given characteristics. X values are sampled randomly and uniformly
    between specified limits. 
    """
    x = np.random.uniform(x_min, x_max, num_points)
    y = amplitude * np.sin(2*np.pi*freq*x + phase)
    return x, y



def random_sinusoid(
    num_metatrain_train, num_metatrain_test,
    num_metavalid_train, num_metavalid_test,
    num_metatest_train, num_metatest_test,
    x_min, x_max,
    freq_min, freq_max,
    amp_min, amp_max,
):
    # phase = 
    pass

def generate_sinusoid_grid():
    pass


if __name__ == "__main__":
    x, y = generate_sinusoid(num_points=10)
    print(x, y)
    print(x.shape)
    plt.plot(x, y, 'bx')
    plt.grid(True)
    plt.savefig("./Code/data/sample_data")