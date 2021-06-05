## this is example skeleton code for a Tensorflow/PyTorch neural network 
## module. You are not required to, and indeed probably should not
## follow these specifications exactly. Just try to get a sense for the kind
## of program structure that might make this convenient to implement.

# overall module class which is subclassed by layer instances
# contains a forward method for computing the value for a given
# input and a backwards method for computing gradients using
# backpropogation.

import numpy as np
from sklearn.utils import shuffle

class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.output = None # output of forward call for backprop.
        self.input = None

    learning_rate = 1E-2 # class-level learning rate

    def __call__(self, input):
        if isinstance(input, Module):
            # todo. chain two networks together with module1(module2(x))
            # update prev and output
            self.prev = input

            if self.prev.output is not None:
                self.output = self.forward(self.prev.output)
        else:
            # todo. evaluate on an input.
            # update output
            self.output = self.forward(input)

        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError


# sigmoid non-linearity
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        # todo. compute sigmoid, update fields
        self.input = input
        y = np.zeros_like(input)
        for i in range(len(input)):
            y[i] = 1/(1 + np.exp(-input[i]))
        
        self.output = y
        return y

    def backwards(self, gradient):
        # todo. compute gradients with backpropogation and data from forward pass
        A = self.output - np.square(self.output)
        A = np.average(A, axis=0)

        D = np.zeros((len(A),len(A)))
        np.fill_diagonal(D,A)

        gradBack = np.matmul(D,gradient)

        return gradBack


# linear (i.e. linear transformation) layer
class Linear(Module):
    def __init__(self, input_size, output_size, is_input=False):
        super(Linear, self).__init__()
        self.W = np.random.rand(input_size,output_size)
        self.b = np.random.rand(output_size)
        self.Adam = Adam(self.W)

        self.is_input = is_input #Currently unused

    def forward(self, input):  # input has shape (batch_size, input_size)
        # todo compute forward pass through linear input
        self.input = input
        y=np.zeros((len(input),len(self.b)))
        for i in range(len(input)):
            y[i] = np.matmul(self.W.T,input[i]) + self.b

        self.output = y
        return y

    def backwards(self, gradient):
        # todo compute and store gradients using backpropogation
        avgIn = np.average(self.input, axis=0)
        gradW = np.outer(avgIn,gradient)

        gradBack = np.matmul(self.W,gradient)

        #For non-Adam gradient descent
        #self.W += -self.learning_rate*gradW
        #self.b += -self.learning_rate*gradient
        gradW = np.average(gradW,axis=0)
        gradb = np.average(gradient,axis=0)

        self.W,self.b = self.Adam(self.W, self.b, gradW, gradb)

        #Can automate by calling prev backwards

        return gradBack


# generic loss layer for loss functions
class Loss:
    def __init__(self):
        self.prev = None

    def __call__(self, input):
        self.prev = input
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()

    def forward(self, input, labels):  # input has shape (batch_size, input_size)
        # todo compute loss, update fields
        self.error = np.zeros((len(input[0])))
        for i in range(len(input)):
            self.error = self.error + (input[i] - labels[i])
        self.error = self.error*(1/len(input))
        return self.error**2

    def backwards(self):
        # todo compute gradient using backpropogation
        return self.error


## overall neural network class
class Network(Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        # todo initializes layers, i.e. sigmoid, linear
        self.lin1 = Linear(input_dim, 5, is_input=True)
        
        self.sig1 = Sigmoid()
        self.sig1(self.lin1)

        self.lin2 = Linear(5,5)
        self.lin2(self.sig1)

        self.sig2 = Sigmoid()
        self.sig2(self.lin2)

        self.lin3 = Linear(5,output_dim)
        self.lin3(self.sig2)

        self.sig3 = Sigmoid()
        self.sig3(self.lin3)

        """

        self.lin4 = Linear(20,output_dim)
        self.lin4(self.sig3)

        self.sig4 = Sigmoid()
        self.sig4(self.lin4)
        """

    def forward(self, input):
        # todo compute forward pass through all initialized layers
        F = self.lin1.forward(input)
        F = self.sig1.forward(F)

        F = self.lin2.forward(F)
        F = self.sig2.forward(F)

        F = self.lin3.forward(F)
        F = self.sig3.forward(F)
        """
        F = self.lin4.forward(F)
        F = self.sig4.forward(F)
        """

        return F

    def backwards(self, grad):
        # todo iterate through layers and compute and store gradients
        """
        G = self.sig4.backwards(grad)
        G = self.lin4.backwards(G)
        """

        G = self.sig3.backwards(grad)
        G = self.lin3.backwards(G)

        G = self.sig2.backwards(G)
        G = self.lin2.backwards(G)

        G = self.sig1.backwards(G)
        G = self.lin1.backwards(G)
        return G

    def predict(self, data):
        # todo compute forward pass and output predictions
        return self.forward(data)

    def accuracy(self, test_data, test_labels):
        # todo evaluate accuracy of model on a test dataset
        preds = self.predict(test_data)
        acc = 0
        for i in range(len(preds)):
            if (preds[i] == test_labels[i]):
                acc += 1
        acc = acc/len(preds)
        return acc


# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations, batch_size, learning_rate=None):
    # todo repeatedly do forward and backwards calls, update weights, do 
    # stochastic gradient descent on mini-batches.

    num_batches = int(len(data)/batch_size)
    if (len(data) % batch_size != 0):
        num_batches += 1
    
    loss = MeanErrorLoss()

    losses = np.zeros((num_iterations))
    for j in range(num_iterations):
        L = 0
        error = 0
        data,labels = shuffle(data,labels)

        for i in range(num_batches):
            start = i*batch_size
            stop = (i+1)*batch_size

            preds = model.forward(data[start:stop])
            L += loss.forward(preds,labels[start:stop])
            model.backwards(loss.backwards())

            error += (np.average(preds-labels[start:stop]))**2
        
        error = error/num_batches
        L = np.average(L)/num_batches
        losses[j] = L
        
        if (j % 100 == 0):
            print("Iteration ", j)
            print("Loss: ", L)
            print("Error: ", error)
            print()

    return model, losses

#Helps a lot somehow
class Adam():
    def __init__(self,W):
        self.learning_rate = 1E-3 #Use 1E-2 for sin
        self.epsilon = 1E-8

        self.m = np.zeros_like(W)
        self.v = np.zeros_like(W)

        self.mb = np.zeros_like(W[0])
        self.vb = np.zeros_like(W[0])

    def __call__(self,W,b,dW,db):
        beta1 = .9
        beta2 = .999

        self.m = beta1*self.m + (1-beta1)*dW
        self.v = beta2*self.v + (1-beta2)*(dW**2)
        W += - self.learning_rate * self.m / (np.sqrt(self.v) + self.epsilon)

        self.mb = beta1*self.mb + (1-beta1)*db
        self.vb = beta2*self.vb + (1-beta2)*(db**2)
        b += - self.learning_rate * self.mb / (np.sqrt(self.vb) + self.epsilon)

        return W,b

        