from train import train2
import numpy as np
from tqdm import tqdm
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def getPlots(kernel, scheme):

    #Get pairs of C and gamma
    Cs = np.arange(.4,2.1,.4)
    Gs = np.arange(.4,2.1,.4)

    A=np.array([0.0,0.0,0.0,0.0,0.0])
    L=np.array([0.0,0.0,0.0,0.0,0.0])

    #Record accuracy and runtime for C values
    i=0
    for c in tqdm(Cs, desc=str(kernel)+str(scheme)+' (C)', leave=False):
        tic=time.perf_counter()
        A[i] = train2(C=c, kernel=kernel, scheme=scheme)
        tok=time.perf_counter()

        L[i]=tok-tic

        i+=1
    
    plt.plot(Cs,A)
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Cacc'
    plt.savefig(path+'.png')
    plt.clf()

    plt.plot(Cs,L)
    plt.xlabel('C')
    plt.ylabel('Runtime')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Crun'
    plt.savefig(path+'.png')
    plt.clf()

    #Repeat for gamma
    i=0
    for g in tqdm(Gs, desc=str(kernel)+str(scheme)+' (gamma)', leave=False):
        tic=time.perf_counter()
        A[i] = train2(gamma=g, kernel=kernel, scheme=scheme)
        tok=time.perf_counter()

        L[i]=tok-tic

        i+=1

    plt.plot(Gs,A)
    plt.xlabel('G')
    plt.ylabel('Accuracy')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Gacc'
    plt.savefig(path+'.png')
    plt.clf()

    plt.plot(Gs,L)
    plt.xlabel('G')
    plt.ylabel('Runtime')
    plt.title('Kernel: '+str(kernel)+', Scheme: '+str(scheme))

    path='digits/'+str(kernel)+'/'+str(kernel)+str(scheme)+'Grun'
    plt.savefig(path+'.png')
    plt.clf()



if __name__=="__main__":
    getPlots('linear','ovo')
    getPlots('linear','ovr')
    
    getPlots(2,'ovo')
    getPlots(2,'ovr')
    
    getPlots(3,'ovo')
    getPlots(3,'ovr')
    
    getPlots(5,'ovo')
    getPlots(5,'ovr')
    
    getPlots('rbf','ovo')
    getPlots('rbf','ovr')