__author__ = 'Wout'

from numpy import *


class FeedForwardNetwork:

    def __init__(self, nIn, nHidden, nOut, rate = 0.001):
        # learning rate
        self.alpha = rate
                                                 
        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nOut = nOut

        self.correct = zeros((self.nOut,1), dtype=float)
         
        # initialize weights randomly (+1 for bias)
        self.hWeights = random.random((self.nHidden, self.nIn+1)) 
        self.oWeights = random.random((self.nOut, self.nHidden+1))
         
        # activations of neurons (sum of inputs)
        self.hActivation = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)
         
        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut), dtype=float)
         
        # deltas for hidden and output layer
        self.hDelta = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)   
     
    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0
         
        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)
         
        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0
         
        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput)
        self.oOutput = tanh(self.oActivation)
     
    def backward(self, teach):
        self.correct[:,0] = teach
        error = self.oOutput - self.correct
        error = self.oOutput - array(teach, dtype=float) 
         
        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation) * tanh(self.oActivation)) * error
                 
        # deltas of hidden neurons
        self.hDelta = (1 - tanh(self.hActivation)) * tanh(self.hActivation) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)
        print 'HDELTA SHAPE', self.hDelta.shape
	print 'iOut shape', self.iOutput.shape
	print 'hWeight shape', self.hWeights.shape         
        # apply weight changes
        self.hWeights = self.hWeights - self.alpha * dot(self.hDelta, self.iOutput.transpose()) 
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput.transpose())

    def predict(self, Sample):
        self.iOutput[:-1, 0] = Sample


        self.iOutput[-1:, 0] = 1.0

        # hidden layer
        self.hActivation = dot(self.hWeights, self.iOutput)
        self.hOutput[:-1, :] = tanh(self.hActivation)
        # self.hOutput[:-1, :] = self.hActivation

        # set bias neuron in hidden layer to 1.0
        self.hOutput[-1:, :] = 1.0

        # output layer
        # print 'oWEIGHT size', self.oWeights.shape
        # print 'hOut size', self.hOutput.shape

        self.oActivation = dot(self.oWeights, self.hOutput)
        self.oOutput = tanh(self.oActivation)
        # self.oOutput = self.oActivation

        return self.oOutput

    def writeResults(self, filename):
        results = file(filename, 'a')
        savetxt(filename, self.hWeights)
        # results.write('oWeights\n')
        # results.write(str(self.oWeights)+'\n')
def func(a,b,c):
    return 3 * a - 2*b + 6*c - 3


if __name__ == '__main__':
    '''
    Tests the network on a simple linear function with 3 variables
    '''
    # define training set
    xorSet = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xorTeach = [[0], [1], [1], [0]]

    # create network
    ffn = FeedForwardNetwork(2,2,1,0.1 )

    for i in range(10000):
        r = random.randint(0,4)
        ffn.forward(xorSet[r])
        ffn.backward(xorTeach[r])

    for i in xorSet:
       
        print 'prediction', i, ffn.predict(i)
        # print 'ERROR', func(a,b,c) - ffn.predict([a,b,c])

    # ffn.writeResults('Simplefuncweights.txt')

