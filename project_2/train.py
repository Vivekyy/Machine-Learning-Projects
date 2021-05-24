from utils import getData
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

def train(dataset, C=1.0, gamma='scale', kernel='rbf', plotpath='Temp'):

    X1,X2,Y = getData(dataset)

    X = np.vstack((X1,X2)).T

    Xtrain = X[:7000]
    Ytrain = Y[:7000]
    Xtest = X[7000:]
    Ytest = Y[7000:]

    if (kernel=='rbf'):
        model = SVC(C=C, kernel=kernel, gamma=gamma)
    elif (kernel==1 or kernel=='linear'):
        model = SVC(C=C, kernel='linear', gamma=gamma)
    else:
        model = SVC(C=C, kernel='poly', degree=kernel, gamma=gamma)
    
    model.fit(Xtrain,Ytrain)

    accuracy = model.score(Xtest,Ytest)

    #Plot decision boundary
    Xbound = np.mgrid[0:1:.01, 0:1:.01].reshape(2,-1).T
    Ybound = model.predict(Xbound)

    title = "Kernel "+str(kernel)+": C = "+str(C)+", Gamma = "+str(gamma)
    plt.scatter(Xbound[:,0], Xbound[:,1], c=Ybound)
    plt.title(title)
    plt.xlabel("Accuracy: "+str(accuracy))

    plt.savefig(plotpath+'.png')
    
    return accuracy

if __name__=="__main__":
    accuracy = train('moons_dataset.csv')
    print("Test Accuracy: ", accuracy)