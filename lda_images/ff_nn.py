__author__ = 'Wout'

from numpy import *


class FeedForwardNetwork:

    def __init__(self, nIn, nHidden, nOut, rate = 0.001):
        # learning rate
        self.alpha = rate
                                                 
        # number of neurons in each layer
        self.nIn = nIn
        self.nHidden = nHidden
        self.nHidden2 = nHidden
        self.nHidden3 = nHidden
        self.nOut = nOut

        self.correct = zeros((self.nOut,1), dtype=float)
         
        # initialize weights randomly (+1 for bias)
        self.hWeights1 = random.random((self.nHidden, self.nIn+1))
        self.hWeights2 = random.random((self.nHidden, self.nHidden+1))
        self.hWeights3 = random.random((self.nHidden, self.nHidden+1))
	# print 'HWEIGHT', self.hWeights
        self.oWeights = random.random((self.nOut, self.nHidden+1))
         
        # activations of neurons (sum of inputs)
        self.hActivation1 = zeros((self.nHidden, 1), dtype=float)
        self.hActivation2 = zeros((self.nHidden, 1), dtype=float)
        self.hActivation3 = zeros((self.nHidden, 1), dtype=float)
        self.oActivation = zeros((self.nOut, 1), dtype=float)
         
        # outputs of neurons (after sigmoid function)
        self.iOutput = zeros((self.nIn+1, 1), dtype=float)      # +1 for bias
        self.hOutput1 = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.hOutput2 = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.hOutput3 = zeros((self.nHidden+1, 1), dtype=float)  # +1 for bias
        self.oOutput = zeros((self.nOut,1), dtype=float)
         
        # deltas for hidden and output layer
        self.hDelta1 = zeros((self.nHidden), dtype=float)
        self.hDelta2 = zeros((self.nHidden), dtype=float)
        self.hDelta3 = zeros((self.nHidden), dtype=float)
        self.oDelta = zeros((self.nOut), dtype=float)   
     
    def forward(self, input):
        # set input as output of first layer (bias neuron = 1.0)
        self.iOutput[:-1, 0] = input
        self.iOutput[-1:, 0] = 1.0
        
	# print 'iOutput', self.iOutput
 
        # hidden layer
        self.hActivation1 = dot(self.hWeights1, self.iOutput)/self.nHidden
        self.hOutput1[:-1, :] = tanh(self.hActivation1)
        self.hOutput1[-1:, :] = 1.0

        self.hActivation2 = dot(self.hWeights2, self.hOutput1)/self.nHidden
        self.hOutput2[:-1, :] = tanh(self.hActivation2)
        self.hOutput2[-1:, :] = 1.0

        self.hActivation3 = dot(self.hWeights3, self.hOutput2)/self.nHidden
        self.hOutput3[:-1, :] = tanh(self.hActivation3)
        self.hOutput3[-1:, :] = 1.0
	#print 'hOutput', self.hOutput
         
        # set bias neuron in hidden layer to 1.0
        # self.hOutput[-1:, :] = 1.0
         
        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput3)/self.nOut
        self.oOutput = tanh(self.oActivation)
     
    def backward(self, teach):
        self.correct[:,0] = teach
        # print 'corrects shape', self.correct.shape
        error = self.oOutput - self.correct

        # print 'error shape', error.shape
         
        # deltas of output neurons
        self.oDelta = (1 - tanh(self.oActivation) * tanh(self.oActivation)) * error
	#print 'oDelta', self.oDelta
                 
        # deltas of hidden neurons
        self.hDelta3 = (1 - tanh(self.hActivation3)* tanh(self.hActivation3)) * dot(self.oWeights[:,:-1].transpose(), self.oDelta)
        self.hDelta2 = (1 - tanh(self.hActivation2)* tanh(self.hActivation2)) * dot(self.hWeights3[:,:-1].transpose(), self.hDelta3)
        self.hDelta1 = (1 - tanh(self.hActivation1)* tanh(self.hActivation1)) * dot(self.hWeights2[:,:-1].transpose(), self.hDelta2)
        # print 'oDelta', self.oDelta.shape
        # print 'HDELTA SHAPE', self.hDelta.shape
        # print 'iOut shape', self.iOutput.shape
        # print 'hWeight shape', self.hWeights.shape
        # apply weight changes
        self.hWeights1 = self.hWeights1 - self.alpha * dot(self.hDelta1, self.iOutput.transpose())
        self.hWeights2 = self.hWeights2 - self.alpha * dot(self.hDelta2, self.hOutput1.transpose())
        self.hWeights3 = self.hWeights3 - self.alpha * dot(self.hDelta3, self.hOutput2.transpose())
        self.oWeights = self.oWeights - self.alpha * dot(self.oDelta, self.hOutput3.transpose())

    
    def predict(self, Sample):
        self.iOutput[:-1, 0] = Sample


        self.iOutput[-1:, 0] = 1.0

        self.hActivation1 = dot(self.hWeights1, self.iOutput)/self.nHidden
        self.hOutput1[:-1, :] = tanh(self.hActivation1)
        self.hOutput1[-1:, :] = 1.0

        self.hActivation2 = dot(self.hWeights2, self.hOutput1)/self.nHidden
        self.hOutput2[:-1, :] = tanh(self.hActivation2)
        self.hOutput2[-1:, :] = 1.0

        self.hActivation3 = dot(self.hWeights3, self.hOutput2)/self.nHidden
        self.hOutput3[:-1, :] = tanh(self.hActivation3)
        self.hOutput3[-1:, :] = 1.0
	#print 'hOutput', self.hOutput

        # set bias neuron in hidden layer to 1.0
        # self.hOutput[-1:, :] = 1.0

        # output layer
        self.oActivation = dot(self.oWeights, self.hOutput3)/self.nOut
        self.oOutput = tanh(self.oActivation)
        # self.oOutput = self.oActivation

        return self.oOutput
   
    def sigmoid(array):
        return map((1/(1+exp(-x))),array)

    def writeResults(self, filename):
        savetxt(filename+'_hweights', self.hWeights)
        savetxt(filename+'_oweights', self.oWeights)
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

    for i in range(100000):
        r = random.random()
        ffn.forward(r)
        ffn.backward(func(r))

    for i in range(10):
       	r = random.random()
        print 'prediction', r, ffn.predict(r), 'correct', func(r)
        # print 'ERROR', func(a,b,c) - ffn.predict([a,b,c])

    # ffn.writeResults('Simplefuncweights.txt')

