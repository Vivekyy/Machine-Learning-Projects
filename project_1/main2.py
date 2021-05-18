import matplotlib.pyplot as plt
import matplotlib
import MLE
import kNN
from tqdm import tqdm
import numpy as np

matplotlib.use('Agg')

runs = 15

#meanMLEacc = 0
meanL1acc = 0
meanL2acc = 0
meanLINFacc = 0

#MLEacc = np.zeros(runs)
L1acc = np.zeros(runs)
L2acc = np.zeros(runs)
LINFacc = np.zeros(runs)

for i in tqdm(range(runs)):
    #MLEacc = MLE.runMLE()
    L1acc[i] = kNN.runKNN(3, norm='l1')
    L2acc[i] = kNN.runKNN(3)
    LINFacc[i] = kNN.runKNN(3, norm='inf')

    #meanMLEacc = meanMLEacc + MLEacc[i]
    meanL1acc = meanL1acc + L1acc[i]
    meanL2acc = meanL2acc + L2acc[i]
    meanLINFacc = meanLINFacc + LINFacc[i]

#meanMLEacc = meanMLEacc/runs
meanL1acc = meanL1acc/runs
meanL2acc = meanL2acc/runs
meanLINFacc = meanLINFacc/runs

#print("Average MLE Accuracy: ", meanMLEacc)
print("Average kNN Accuracy (L1): ", meanL1acc)
print("Average kNN Accuracy (L2): ", meanL2acc)
print("Average kNN Accuracy (L-inf): ", meanLINFacc)

x = np.arange(runs)
#plt.plot(x, MLEacc, label='MLE')
plt.plot(x, L1acc, label='L1')
plt.plot(x, L2acc, label='L2')
plt.plot(x, LINFacc, label='L-inf')
plt.ylabel('Accuracy')
plt.title("KNN Accuracies")
plt.legend()
plt.savefig('Stats2.png')