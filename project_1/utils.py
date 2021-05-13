import scipy.io as sio
import numpy as np

def getData():
    data = sio.loadmat('digits.mat')
    X = data['X']
    Y = data['Y']

    #shuffle indices
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    return X,Y