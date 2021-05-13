import numpy as np
from tqdm import tqdm
from scipy import stats
from utils import getData

#Estimate class priors
def getPriors(Y):
    i = 0
    zeroprior = 0
    oneprior = 0
    twoprior = 0
    threeprior = 0
    fourprior = 0
    fiveprior = 0
    sixprior = 0
    sevenprior = 0
    eightprior = 0
    nineprior = 0
    while (i<len(Y)):
        if (Y[i]==0):
            zeroprior+=1
        if (Y[i]==1):
            oneprior+=1
        if (Y[i]==2):
            twoprior+=1
        if (Y[i]==3):
            threeprior+=1
        if (Y[i]==4):
            fourprior+=1
        if (Y[i]==5):
            fiveprior+=1
        if (Y[i]==6):
            sixprior+=1
        if (Y[i]==7):
            sevenprior+=1
        if (Y[i]==8):
            eightprior+=1
        if (Y[i]==9):
            nineprior+=1
        i += 1
    zeroprior = zeroprior/len(Y)
    oneprior = oneprior/len(Y)
    twoprior = twoprior/len(Y)
    threeprior = threeprior/len(Y)
    fourprior = fourprior/len(Y)
    fiveprior = fiveprior/len(Y)
    sixprior = sixprior/len(Y)
    sevenprior = sevenprior/len(Y)
    eightprior = eightprior/len(Y)
    nineprior = nineprior/len(Y)

    return [zeroprior, oneprior, twoprior, threeprior, fourprior, fiveprior, sixprior, sevenprior, eightprior, nineprior]

def getConditionals(X,Y):
    means = np.zeros((10, np.shape(X[1])[0]))
    covs = np.zeros((10,np.shape(X[1])[0],np.shape(X[1])[0]))
    for i in range(10):
        means[i], covs[i] = kConditional(X,Y,i)
    
    return means, covs

def kConditional(X,Y,k):

    #Create array with only k valued x
    kArray = np.zeros(np.shape(X))
    numx = 0
    for ind in range(len(X)):
        if (Y[ind]==k):
            kArray[numx] = X[ind]
            numx += 1

    #Compute mean
    mean = np.zeros(np.shape(X[1]))
    for i in range(numx):
        mean += kArray[i]
    mean = mean/numx

    #Compute cov
    cov = np.zeros((len(mean),len(mean)))
    for i in range(numx):
        A = kArray[i] - mean
        cov = cov + np.outer(A, A)
    cov = cov/numx

    return mean, cov

def runMLE():
    #Train on 8000, test on 2000
    X, Y = getData()
    Xtrain = X[:7000]
    Ytrain = Y[:7000]
    Xtest = X[7000:]
    Ytest = Y[7000:]

    means, covs = getConditionals(Xtrain,Ytrain)
    priors = getPriors(Ytrain)

    acc = 0

    Probs = np.zeros((10,len(Xtest)))
    for j in tqdm(range(10), leave=False, desc='MLE'):
        #Fix for non-invertible matrices
        A = covs[j]
        A = A + .01*np.identity(np.shape(covs[j])[0])

        #Use logpdf to avoid overflow
        p = stats.multivariate_normal.logpdf(Xtest, mean=means[j], cov=A)
        p = p*priors[j]
        Probs[j] = p
    ypred = np.zeros(len(Xtest))
    for i in range(len(Xtest)):
        ypred = np.argmax(Probs[:,i])
        if (ypred == Ytest[i]):
            acc += 1
    acc = acc/len(Xtest)

    return acc

if __name__ == "__main__":
    acc = runMLE()
    print("Final Accuracy: ", acc)