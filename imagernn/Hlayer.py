__author__ = 'Thijs'
import numpy as np

from imagernn.utils import initw


class HLayer:
    """
  A hidden layer with a memory cell attached.
  """

    def __init__(self, hidden_size,drop_prob_decoder,predict_mode, id):
        self.id = id
        self.drop_prob_decoder = drop_prob_decoder
        self.predict_mode = predict_mode
        self.d = hidden_size
        self.N = 8  # TODO wijzigen zodat niet gehardcodeded

    def forward(self,H,n,model,cache):
        H2 = initw(n, self.d)
        A = model['A'+str(self.id)]
       # print(A)
       # print(str(A.size))
        M = fillMMatrix(A, n ,self.N)
        #M = np.zeros((n,n))
        #print(M)
        #U2=0
        '''
        if self.drop_prob_decoder > 0: # if we want dropout on the decoder
            if not self.predict_mode: # and we are in training mode
                scale2 = 1.0 / (1.0 - self.drop_prob_decoder)
                U2 = (np.random.rand(*(H.shape)) < (1 - self.drop_prob_decoder)) * scale2 # generate scaled mask
                H *= U2 # drop!
        '''
        # Update Hmem
        Hmem = np.maximum(M.dot(H), 0)

        # Hidden Layer 2
        for t in xrange(n):
            mem = Hmem[t - 1]
            H2[t] = np.maximum(H[t].dot(model['Whh'+str(self.id)]) + mem + model['bhh'+str(self.id)], 0)

        if not self.predict_mode:
            cache['Whh'+str(self.id)] = model['Whh'+str(self.id)]
            cache['H'+str(self.id)] = H2
            cache['bhh'+str(self.id)] = model['bhh'+str(self.id)]
            cache['M'+str(self.id)] = M
            cache['Hmem'+str(self.id)] = Hmem
            #cache['U2']= U2
        return H2,cache,M

    def backward(self,cache,D):
        # backprop H2
        Hid = str(self.id-1) if self.id-1>=0 else ''
        H = cache['H'+Hid]
        Whh = cache['Whh'+str(self.id)]
        M = cache['M'+str(self.id)]
        '''drop_prob_decoder = cache['drop_prob_decoder']'''

        dWhh = H.transpose().dot(D)
        dHmem = D

        # update M en daarmee ook A
        dM = dHmem.dot(H.transpose())
        #print('Shape van dM', dM.shape)
        #print('dM:', dM)
        dA = fromMtoA(dM,self.N,M.shape[0])
        #print(dA)

        # backprop H
        dH = D.dot(Whh.transpose()) + M.transpose().dot(D)
        dbhh = np.sum(D, axis=0, keepdims = True)
        # backprop dropout, if it was applied
        '''if drop_prob_decoder > 0:
            dH *= cache['U2']
        '''
        return{'Whh'+str(self.id):dWhh, 'A'+str(self.id):dA, 'bhh'+str(self.id):dbhh}, dH


def fromMtoA(M,N,n):
    A = np.zeros((1,N))
    for i in range(0,min(N,n)):
        d = np.diag(M,i)
        avg = np.mean(d)
        A[0,i] = avg
    return A


def fillMMatrix(A, n, N):
    ''' Fills the matrix M with the values of vector A, as proposed by Zhang '''
    M = np.zeros((n, n))
    for i in range(0, n):  # Row
        for j in range(0, N):  # Column
            k = i + j
            if k < n:
                M[i, k] = A[0, j]
    return M.transpose()
