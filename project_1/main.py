import matplotlib.pyplot as plt
import matplotlib
import MLE
from tqdm import tqdm
import numpy as np

matplotlib.use('Agg')

meanacc = 0
MLEacc = np.zeros(100)
for i in tqdm(range(100)):
    MLEacc[i] = MLE.runMLE()
    meanacc = meanacc + MLEacc[i]
meanacc = meanacc/100
print(meanacc)

x = np.arange(100)
plt.plot(x, MLEacc)
plt.ylabel('Accuracy')
plt.title("MLE Accuracies")
plt.savefig('MLE_Stats.png')