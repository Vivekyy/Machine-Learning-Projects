import numpy as np

def getData(fileName):
    
    data = np.loadtxt(fileName, delimiter=',', usecols=(0,1,2), unpack=True)
    
    Y=data[0,:]
    X1=data[1,:]
    X2=data[2,:]

    return X1,X2,Y
