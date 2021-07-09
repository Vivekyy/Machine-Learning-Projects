import sklearn.neighbors as neigh
import sklearn.utils.graph_shortest_path as gsp
import sklearn.preprocessing as prep
import numpy as np
from tqdm import tqdm

def redux(X, name):
    graph = neigh.kneighbors_graph(X,10)
    pie = gsp.graph_shortest_path(graph)
    
    Y = np.random.rand(len(X),2)*100
    derivs = np.zeros_like(Y)

    step = .00005 #Need step size to be very small to avoid diverging to NaN values
    for _ in tqdm(range(500), leave=False, desc=name):
        for i in range(len(X)):
            temp = np.ones_like(Y)*Y[i]
            A = np.subtract(temp,Y)
            
            normedA = prep.normalize(A) #Scikit handles the norm (0,0) case by just returning (0,0)
            pies = pie[i,:]
            
            deriv = np.sum(A,axis=0)
            deriv = deriv - np.matmul(pies.T,normedA)

            derivs[i] = deriv

        Y = Y - step*derivs
    
    return Y
            