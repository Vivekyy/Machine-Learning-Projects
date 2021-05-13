import matplotlib.pyplot as plt
import matplotlib
import MLE
import kNN
from tqdm import tqdm
import numpy as np

matplotlib.use('Agg')

runs = 20

meanMLEacc = 0
meanKNNacc = 0
MLEacc = np.zeros(runs)
KNNacc = np.zeros(runs)
for i in tqdm(range(runs)):
    MLEacc[i] = MLE.runMLE()
    KNNacc[i] = kNN.runKNN(4)
    meanMLEacc = meanMLEacc + MLEacc[i]
    meanKNNacc = meanKNNacc + KNNacc[i]
meanMLEacc = meanMLEacc/runs
meanKNNacc = meanKNNacc/runs

print("Average MLE Accuracy: ", meanMLEacc)
print("Average kNN Accuracy: ", meanKNNacc)

x = np.arange(runs)
plt.plot(x, MLEacc, label='MLE')
plt.plot(x, KNNacc, label='kNN')
plt.ylabel('Accuracy')
plt.title("MLE Accuracies")
plt.legend()
plt.savefig('Stats.png')