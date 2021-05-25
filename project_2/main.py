from train import train
import numpy as np
from tqdm import tqdm

def getPlots(kernel):

    #Get pairs of C and gamma
    Cs = np.array([.5,1,5])
    Gs = np.array([.5,1,5])

    C,G = np.meshgrid(Cs,Gs)
    CG = np.vstack((C.flatten(),G.flatten())).T

    f = open('acc.txt', 'a')
    f.write(str(kernel)+'\n')
    f.write('\n')
    for cg in tqdm(CG, desc='Kernel '+str(kernel), leave=False):
        c=cg[0]
        g=cg[1]

        path = 'moons/'+str(kernel)+'/MK'+str(kernel)+'C'+str(c)+'G'+str(g)
        Macc,Mtrainacc = train('moons_dataset.csv', C=c, gamma=g, kernel=kernel, plotpath=path)


        path = 'rolls/'+str(kernel)+'/RK'+str(kernel)+'C'+str(c)+'G'+str(g)
        Racc,Rtrainacc = train('rolls_dataset.csv', C=c, gamma=g, kernel=kernel, plotpath=path)

        f.write(str(cg)+'\n')
        f.write("Moons Train Accuracy: %s \n" % str(Mtrainacc))
        f.write("Moons Test Accuracy: %s \n" % str(Macc))
        if(Mtrainacc>Macc):
            f.write("Possible Overfit \n")
        f.write('\n')
        f.write("Rolls Train Accuracy: %s \n" % str(Rtrainacc))
        f.write("Rolls Test Accuracy: %s \n" % str(Racc))
        if(Rtrainacc>Racc):
            f.write("Possible Overfit \n")
        f.write('\n')
        f.write('\n')
    f.close()

if __name__=="__main__":
    getPlots('linear')
    getPlots(2)
    getPlots(3)
    getPlots(5)
    getPlots('rbf')