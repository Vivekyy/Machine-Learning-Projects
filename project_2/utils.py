import numpy as np
from sklearn import preprocessing
import scipy.io as sio

def getData(fileName):
    
    data = np.loadtxt(fileName, delimiter=',', usecols=(0,1,2), unpack=True)
    
    Y=data[0,:]
    X1=data[1,:]
    X2=data[2,:]

    #shuffle indices
    shuffler = np.random.permutation(len(X1))
    X1 = X1[shuffler]
    X2 = X2[shuffler]
    Y = Y[shuffler]

    return X1,X2,Y

def getDigits():
    data = sio.loadmat('digits.mat')
    X = data['X']
    Y = data['Y']

    #shuffle indices
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    return X,Y