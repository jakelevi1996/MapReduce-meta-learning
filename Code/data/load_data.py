import numpy as np
from generate_data import DEFAULT_FILENAME

def load_sinusoidal_data(filename=DEFAULT_FILENAME):
    with np.load(filename) as data:
        X_metatrain = data["X_metatrain"]
        Y_metatrain = data["Y_metatrain"]
        X_metavalid = data["X_metavalid"]
        Y_metavalid = data["Y_metavalid"]
        X_metatest = data["X_metatest"]
        Y_metatest = data["Y_metatest"]
    
    return X_metatrain, Y_metatrain,    \
    X_metavalid, Y_metavalid,           \
    X_metatest, Y_metatest