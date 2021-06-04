#Test on plotting quadratic curve

import numpy as np
from net import Network,train
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

X = np.random.rand(1000,1)
y = X**2

net = Network(1,1)
net = train(net,X,y,200,10)

plt.scatter(net.predict(X),y)
plt.savefig('fig.png')