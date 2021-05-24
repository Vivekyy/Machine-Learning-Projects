from train import train
import numpy as np
from tqdm import tqdm

def getPlots(kernel):

    #Get pairs of C and gamma
    Cs = np.array([.5,1,5])
    Gs = np.array([.5,1,5])

    C,G = np.meshgrid(Cs,Gs)
    CG = np.vstack((C.flatten(),G.flatten())).T

    for cg in tqdm(CG, desc='Kernel '+str(kernel), leave=False):
        c=cg[0]
        g=cg[1]

        path = 'moons/'+str(kernel)+'/MK'+str(kernel)+'C'+str(c)+'G'+str(g)
        train('moons_dataset.csv', C=c, gamma=g, kernel=kernel, plotpath=path)


        path = 'rolls/'+str(kernel)+'/RK'+str(kernel)+'C'+str(c)+'G'+str(g)
        train('rolls_dataset.csv', C=c, gamma=g, kernel=kernel, plotpath=path)

if __name__=="__main__":
    #getPlots('linear')
    #getPlots(2)
    #getPlots(3)
    getPlots(5)
    #getPlots('rbf')