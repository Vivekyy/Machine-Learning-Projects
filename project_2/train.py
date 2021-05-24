from utils import getData
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def train(dataset, C=1.0, gamma='scale', kernel='rbf', plotname='Temp'):

    X1,X2,Y = getData(dataset)

    X = np.vstack((X1,X2)).T

    Xtrain = X[:8000]
    Ytrain = Y[:8000]
    Xtest = X[8000:]
    Ytest = Y[8000:]

    if (kernel=='rbf'):
        model = SVC(C=C, kernel=kernel, gamma=gamma)
    else:
        model = SVC(C=C, kernel='poly', degree=kernel, gamma=gamma)
    
    model.fit(Xtrain,Ytrain)

    accuracy = model.score(Xtest,Ytest)

    #Plot decision boundary
    Xbound = np.mgrid[0:1:.01, 0:1:.01].reshape(2,-1).T
    Ybound = model.predict(Xbound)

    plt.scatter(Xbound[:,0], Xbound[:,1], c=Ybound)
    plt.title(plotname)
    plt.savefig(plotname+'.png')

    return accuracy

if __name__=="__main__":
    train('moons_dataset.csv')