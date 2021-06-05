#Test on plotting quadratic curve

import numpy as np
from net import Network,train
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

iterations = 500

X = np.random.rand(1000,1)*2*np.pi
y = (1+np.sin(X))/2

net = Network(1,1)
net,loss = train(net,X,y,iterations,10)

plt.scatter(X,net.predict(X))
plt.xlabel('X')
plt.ylabel('Predicted y: sin(x)')
plt.savefig('sin.png')

plt.clf()
plt.plot(np.arange(iterations),loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Sin Approximation Loss')
plt.savefig('sinloss.png')