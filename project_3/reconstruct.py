#Configure network to reconstruct two provided images (flower,Lincoln)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import getData
from net import Network,train

matplotlib.use('Agg')

#Flower
iterations = 2000

X,y = getData(2)
netF = Network(2,3)
netF,loss = train(netF,X,y,iterations,100)

preds = netF.predict(X)
plt.imshow(preds.reshape(133,140,3))
plt.title('Flower')
plt.savefig('Flower.png')

plt.clf()
plt.plot(np.arange(iterations),loss)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title('Flower Loss')
plt.savefig('Flowerloss.png')

#Lincoln
iterations = 2000
X,y = getData(1)
netL = Network(2,1)
netL,loss = train(netL,X,y,iterations,100)

preds = netL.predict(X)
plt.imshow(preds.reshape(100,76))
plt.title('Lincoln')
plt.savefig('Lincoln.png')

plt.clf()
plt.plot(np.arange(iterations),loss)
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title('Lincoln Loss')
plt.savefig('Lincolnloss.png')