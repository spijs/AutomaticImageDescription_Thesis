import numpy as np


class StackedCCAModel:
    def __init__(self, nn_img, nn_sent, cca, augmentedcca):
        self.nn_img = nn_img
        self.nn_sent = nn_sent
        self.cca = cca
        self.augmentedcca = augmentedcca

    def getTransformedVector(self, imageVector):
        firstProjection = self.cca.transform(imageVector.reshape(1,-1))
        augmentedVector = np.append(firstProjection, self.phi(3000 - len(firstProjection), self.nn_img, firstProjection))
        return self.augmentedcca.transform(augmentedVector.reshape(1,-1))

    def phi(wantedDimension, sigma, x):
        b = np.random.rand(wantedDimension)
        R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
        return np.dot(x,R) + b