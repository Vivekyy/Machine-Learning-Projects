from utils import getData
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

X1,X2,Y = getData('moons_dataset.csv')

plt.scatter(X1, X2, c=Y)
plt.title("Moons")
plt.savefig('Moons.png')

X1,X2,Y = getData('rolls_dataset.csv')

plt.scatter(X1, X2, c=Y)
plt.title("Rolls")
plt.savefig('Rolls.png')