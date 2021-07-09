import numpy as np
from dim import redux
import sklearn.decomposition as decomp
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

def getData(name):
    f = open(name,'r')
    X = []
    for line in f:
        X.append([float(x) for x in line.split()])
    f.close()

    X = np.reshape(X,(-1,3))
    return X     

def run(data,name):

    #Plot original data
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2])
    plt.title('Original Data for %s' % name)
    plt.savefig(name+".png")
    plt.clf()

    Y = redux(data,name)

    plt.scatter(Y[:,0],Y[:,1])
    plt.title('PCA Data for %s' % name)
    plt.savefig(name+"nl.png")
    plt.clf()

    data1 = data
    PCA = decomp.PCA(n_components=2)
    PCA.fit(data1) #After redux because this alters data

    plt.scatter(data1[:,0],data1[:,1])
    plt.title('PCA Data for %s' % name)
    plt.savefig(name+"pca.png")
    plt.clf()


if __name__ == "__main__":
    X1 = getData('swiss_roll.txt')
    X2 = getData('swiss_roll_hole.txt')

    run(X1,"SwissRoll")
    run(X2,"SwissRollHole")