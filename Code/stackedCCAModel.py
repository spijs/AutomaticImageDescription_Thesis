import numpy as np


class StackedCCAModel:
    def __init__(self, nn_img, nn_sent, cca, augmentedcca):
        self.nn_img = nn_img
        self.nn_sent = nn_sent
        self.cca = cca
        self.augmentedcca = augmentedcca

    def getTransformedVector(self, imageVector):
        firstProjection = self.cca.transform(imageVector.reshape(1,-1))
        augmentedVector = np.append(firstProjection[0], self.phi((3000 - len(firstProjection[0])), self.nn_img, firstProjection[0]))
        return self.augmentedcca.transform(augmentedVector.reshape(1,-1))

    def phi(self,wantedDimension, sigma, x):
        b = np.random.rand(wantedDimension)
	print "X: " + str(x)
	print str(b.shape)
        R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
	print str(R.shape)
	print str(x.shape)
        return np.dot(x,R) + b
