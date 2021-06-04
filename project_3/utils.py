import scipy.io as sio
import numpy as np
from sklearn import preprocessing

def getData(in):
    data = sio.loadmat('nn_data.mat')
    X = data['X'+str(in)]
    Y = data['Y'+str(in)]

    #normalize X data
    X = preprocessing.normalize(X, norm='max', axis=1)
    Y = Y/255 #Pixel data is 0-255

    #shuffle indices
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    return X,Y