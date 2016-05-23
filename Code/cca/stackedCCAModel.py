import numpy as np


class StackedCCAModel:
    def __init__(self, nn_img, nn_sent, cca, augmentedcca):
        '''

        :param nn_img: average distance to 50th nearest neighbor of the images
        :param nn_sent: average distance to 50th nearest neighbor of the sentences
        :param cca: cca model used for the first projection
        :param augmentedcca: cca model used for the second projection
        '''
        self.nn_img = nn_img
        self.nn_sent = nn_sent
        self.cca = cca
        self.augmentedcca = augmentedcca

    def getTransformedVector(self, imageVector):
        '''
        Calculate the transformed version of the given image vector
        :param imageVector
        :return: augmented version of the given vector, using Stacked Auxiliary Embedding
        '''
        firstProjection = self.cca.transform(imageVector.reshape(1,-1))
        augmentedVector = np.append(firstProjection[0], self.phi((3000 - len(firstProjection[0])), self.nn_img, firstProjection[0]))
        return self.augmentedcca.transform(augmentedVector.reshape(1,-1))

    def phi(self,wantedDimension, sigma, x):
        '''
        Random fourier feature function based on a given sigma, applied to augment x to the wantedDimension
        :param wantedDimension:
        :param sigma:
        :param x:
        :return: application of RFF function to the given x, to augment it to the wanted dimension
        '''
        b = np.random.rand(wantedDimension)
        R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
        return np.dot(x,R) + b
