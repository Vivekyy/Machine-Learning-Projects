#Test on plotting quadratic curve

import numpy as np
from net import Network,train
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

iterations = 1000

X = np.random.rand(1000,1)
y = X**2

net = Network(1,1)
net,loss = train(net,X,y,iterations,20)

plt.scatter(X,net.predict(X))
plt.xlabel('X')
plt.ylabel('Predicted y: x^2')
plt.savefig('quad2.png')

plt.clf()
plt.plot(np.arange(iterations),loss)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Quadratic Approximation Loss')
plt.savefig('quadloss2.png')