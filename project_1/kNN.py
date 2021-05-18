#from scipy.spatial import distance
#Scipy distance is roughly twice as slow as numpy norm
import numpy as np
import math
from utils import getData
from tqdm import tqdm

def runKNN(k, norm='l2'):
    X, Y = getData()
    Xtrain = X[:8000]
    Ytrain = Y[:8000]
    Xtest = X[8000:]
    Ytest = Y[8000:]

    description = "KNN "
    if (norm == 'l2'):
        description += " (L2)"
    elif (norm == 'l1'):
        description += " (L1)"
    elif (norm == 'inf'):
        description += " (L-inf)"

    acc = 0
    for i in tqdm(range(len(Xtest)), leave=False, desc=description):
        ypred = getAssignment(Xtest[i], k, Xtrain, Ytrain, norm)
        if (ypred == Ytest[i]):
            acc += 1
    acc = acc/len(Xtest)

    return acc

def getAssignment(x, k, Xtrain, Ytrain, norm):
    #Collect k-best
    kBest = np.zeros(k, dtype='int64')
    
    test = 0
    tempN = math.inf*np.ones((k,2))
    for j in range(len(Xtrain)):
        #Calculate d based on norm input
        d=0
        if (norm=='l1'):
            d=np.linalg.norm(Xtrain[j]-x, ord=1)
        elif (norm=='l2'):
            d=np.linalg.norm(Xtrain[j]-x)
        elif (norm=='inf'):
            d=np.linalg.norm(Xtrain[j]-x, ord=math.inf)

        #Update array of distances/values if new better value found
        if (d<tempN[k-1,0]):
            #Make new array
            tempA = math.inf*np.ones((k+1,2))
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
    acc = runKNN(3, norm='l1')
    print(acc)