__author__ = 'Thijs'

import unittest
import numpy as np
from imagernn.Hlayer import fillMMatrix,fromMtoA

class FSMN_test(unittest.TestCase):

    ''' Test if fillmatrix behaves as expected under normal circumstances'''
    def test_fillMatrix_normal(self):
        A = np.array([[1,2,3,4]])
        N = len(A)
        n = 6
        T = fillMMatrix(A,n,N)
        T = T.transpose()
        for i in range(0,N):
            self.assertEqual(T[0,i],A[0,i])
        for i in range(N,n):
            self.assertEqual(T[0,i],0)
        for i in range(0,N):
            self.assertEqual(T[1,i+1],A[0,i])
        for i in range(N+1,n):
            self.assertEqual(T[1,i],0)
        self.assertEqual(T[1,0],0)

    def test_fromMtoA_normal(self):
        M = [[1,2,3,4,0],
             [0,0,1,2,0],
             [0,0,2,3,1],
             [1,2,3,4,5],
             [0,2,3,4,5]]
        N = 3
        A = fromMtoA(M,N)
        self.assertEqual(A[0,0],(sum(np.diag(M,0)*1.0)/5))
        self.assertEqual(A[0,1],1.0*(2+1+3+5)/4)
        self.assertEqual(A[0,2],1.0*(3+2+1)/3)
        self.assertEqual(A.shape,(1,3))


if __name__ == '__main__':
    unittest.main()
