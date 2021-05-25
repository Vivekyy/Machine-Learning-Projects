from train import train2
import numpy as np
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def getPlots(kernel, scheme):

    #Get pairs of C and gamma
    Ns = np.array([100,300,500,700])

    A=np.array([0.0,0.0,0.0,0.0])
    L=np.array([0.0,0.0,0.0,0.0])

    #Record accuracy and runtime for C values
    i=0
    for n in tqdm(Ns, desc=str(kernel)+str(scheme)+' (reduc)', leave=False):
        tic=time.perf_counter()
        A[i] = train2(kernel=kernel, scheme=scheme, n=n)
        tok=time.perf_counter()

        L[i]=tok-tic

        i+=1
    
    plt.plot(Ns,A)
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Nacc'
    plt.savefig(path+'.png')
    plt.clf()

    plt.plot(Ns,L)
    plt.xlabel('N')
    plt.ylabel('Runtime')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Nrun'
    plt.savefig(path+'.png')
    plt.clf()

if __name__=="__main__":   
    getPlots(2,'ovo')
    getPlots(2,'ovr')
    
    getPlots(3,'ovo')
    getPlots(3,'ovr')
    
    getPlots(5,'ovo')
    getPlots(5,'ovr')
    
    getPlots('rbf','ovo')
    getPlots('rbf','ovr')