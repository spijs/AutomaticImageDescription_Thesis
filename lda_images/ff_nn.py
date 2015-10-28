__author__ = 'Wout'

from numpy import *
from imagernn.utils import initw
import time

class FeedForwardNetwork:

    def __init__(self, nIn, nHidden, nOut, rate = 0.001):
        # learning rate
        self.alpha = rate

        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut

        # initialize weights randomly (+1 for bias)
        self.activation1 = zeros((self.nHidden, 1), dtype=float)
        self.weights1 = initw(self.nHidden+1, self.nIn+1)
        self.input = zeros((self.nIn+1, 1), dtype=float)
        self.activation2 = zeros((self.nOut, 1), dtype= float)
        self.weights2 = initw(self.nOut, self.nHidden+1)

    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.input[:-1, 0] = input
        self.input[-1:, 0] = 1.0

        # hidden layer
        self.activation1 = dot(self.weights1, self.input)
        self.hOutput1 = tanh(self.activation1)

        self.activation2 = dot(self.weights2, self.hOutput1)
        e = exp(self.activation2)
        self.oOutput = e / sum(e)

    def backward(self, teach):
        self.correct = zeros((self.nOut,1), dtype=float)
        self.correct[:,0] = teach
        error = self.oOutput - self.correct

        # deltas of output neuron
        oDelta = self.alpha * error
        # delta of hidden neuron
        hDelta = (1-tanh(self.activation1))*tanh(self.activation1)*dot(self.weights2.transpose(),oDelta)

        #apply weight changes
        self.weights1 = self.weights1 - dot(hDelta,self.input.transpose())
        self.weights2 = self.weights2 - dot(oDelta,self.hOutput1.transpose())


    def cost(self):
        prediction = self.oOutput
        #print('log van prediction', log(prediction))
        #time.sleep(1.0)
        cost = -sum(self.correct*log(prediction))
        return cost

    def predict(self, input):
     # set input as output of first layer (bias neuron = 1.0)
        self.input[:-1, 0] = input
        self.input[-1:, 0] = 1.0

        # hidden layer
        self.activation1 = dot(self.weights1, self.input)
        self.hOutput1 = tanh(self.activation1)

        self.activation2 = dot(self.weights2, self.hOutput1)
        e = exp(self.activation2)
        self.oOutput = e / sum(e)

    def sigmoid(array):
        return map((1/(1+exp(-x))),array)

    def writeResults(self, filename):
        savetxt(filename+'_hweights', self.weights1)
        #savetxt(filename+'_oweights', self.oWeights)
        # results.write('oWeights\n')
        # results.write(str(self.oWeights)+'\n')
def func(a):
    return [abs(cos(a)), abs(sin(a))]


if __name__ == '__main__':
    '''
    Tests the network on a simple linear function with 3 variables
    '''
    # define training set
    xorSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xorTeach = [[0,0], [1,1], [1,1], [0,0]]

    # create network
    ffn = FeedForwardNetwork(1,2,2,0.01 )

    for i in range(10):
        r = random.random()
        ffn.forward(r)
        ffn.backward(func(r))

    for i in range(10):
        r = random.random()
        print 'prediction', r, ffn.predict(r), 'correct', func(r)
        # print 'ERROR', func(a,b,c) - ffn.predict([a,b,c])

        # ffn.writeResults('Simplefuncweights.txt')

