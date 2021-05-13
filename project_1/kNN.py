import math
import numpy as np
from utils import getData
from tqdm import tqdm

def runKNN(k, norm=np.linalg.norm):
    X, Y = getData()
    Xtrain = X[:7000]
    Ytrain = Y[:7000]
    Xtest = X[7000:]
    Ytest = Y[7000:]

    acc = 0
    for i in tqdm(range(len(Xtest)), leave=False, desc="kNN"):
        ypred = getAssignment(Xtest[i], k, Xtrain, Ytrain, norm)
        if (ypred == Ytest[i]):
            acc += 1
    acc = acc/len(Xtest)

    return acc

def getAssignment(x, k, Xtrain, Ytrain, norm):
    #Collect k-best
    kBest = np.zeros(k, dtype='int64')
    
    test = 0
    tempN = 10000*np.ones((k,2))
    for j in range(len(Xtrain)):
        d = norm(Xtrain[j]-x)

        #Update array of distances/values if new better value found
        if (d<tempN[k-1,0]):
            #Make new array
            tempA = 10000*np.ones((k+1,2))
            for i in range(k):
                tempA[i] = tempN[i]
            tempA[k,0] = d
            tempA[k,1] = Ytrain[j]
            
            #Sort by first column values (distances)
            tempA = tempA[np.argsort(tempA[:,0])]

            for i in range(k):
                tempN[i] = tempA[i]
    
    for i in range(k):
        kBest[i] = tempN[i,1]
            
    vals = np.bincount(kBest)
    assignment = np.argmax(vals)

    return assignment

if __name__ == "__main__":
    acc = runKNN(4)
    print(acc)