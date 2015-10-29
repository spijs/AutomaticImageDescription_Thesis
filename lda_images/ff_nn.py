__author__ = 'Wout'

from numpy import *
from imagernn.utils import initw

class FeedForwardNetwork:

    def __init__(self, nIn, nHidden, nOut, rate = 0.001):
        # learning rate
        self.alpha = rate

        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut

        self.correct = zeros((self.nOut,1), dtype=float)
        self.hweights = initw(self.nHidden,self.nIn+1)
        self.weights = initw(self.nOut, self.nHidden+1)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)
        self.hOutput = zeros((self.nHidden+1,1),dtype=float)
        self.activation = zeros((self.nOut, 1), dtype= float)
        self.hactivation = zeros((self.nHidden,1),dtype=float)
        self.oDelta = zeros((self.nOut,1), dtype=float)

    def forward(self, input):

        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0

        #Hidden
        self.hactivation = dot(self.hweights,self.iOutput)
        #self.hOutput[:-1,:] = tanh(self.hactivation)
        self.hOutput[:-1, :] = self.hactivation
        self.hOutput[-1:,0] = 1.0

        #Single layer
        self.activation = dot(self.weights, self.hOutput)
        e = exp(self.activation)
        self.oOutput = e / sum(e)

    def backward(self, teach):
        self.correct[:,0] = teach
        error = self.oOutput - self.correct
        self.oDelta = self.alpha * error

        # deltas of hidden neurons
        #self.hDelta = (1 - tanh(self.hactivation)) * tanh(self.hactivation) * dot(self.weights[:,:-1].transpose(), self.oDelta)
        self.hDelta = dot(self.weights[:,:-1].transpose(),self.oDelta)

        self.weights = self.weights - dot(self.oDelta, self.hOutput.transpose())

        # apply weight changes
        self.hweights = self.hweights - self.alpha * dot(self.hDelta, self.iOutput.transpose())


    def cost(self):
        prediction = self.oOutput
        cost = -sum(self.correct*log(prediction))
        return cost

    def predict(self, Sample):
        self.iOutput[:-1, 0] = Sample
        self.iOutput[-1:, 0] = 1.0

        #Hidden
        self.hactivation = dot(self.hweights,self.iOutput)
        # self.hOutput[:-1, :] = tanh(self.hactivation) met tanh
        self.hOutput[:-1, :] = self.hactivation
        self.hOutput[-1:,0] = 1.0

        #Single layer
        self.activation = dot(self.weights, self.hOutput)
        e = exp(self.activation)
        self.oOutput = e / sum(e)
        return self.oOutput.flatten()


    def writeResults(self, filename):
        #savetxt(filename+'_hweights', self.weights)
        savetxt(filename+'_oweights', self.weights)
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
