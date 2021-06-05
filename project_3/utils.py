import scipy.io as sio
import numpy as np
from sklearn import preprocessing

def getData(dataset):
    data = sio.loadmat('nn_data.mat')
    X = data['X'+str(dataset)]
    Y = data['Y'+str(dataset)]

    #normalize X data
    X = preprocessing.normalize(X, norm='max', axis=1)
    Y = Y/255 #Pixel data is 0-255

    return X,Y