from sknn.mlp import MultiLayerPerceptron

__author__ = 'Wout & thijs'
from imagernn.data_provider import getDataProvider
from lda_images.ff_nn import *
import numpy as np
from sknn.mlp import *
import copy
import sys

class LDANetworkLearner:


    def __init__(self, dataset, nbOfTopics, rate,hidden,layers):
        self.nbOfTopics=nbOfTopics
        self.dataset = dataset
        self.dataprovider = getDataProvider(dataset)

        # self.network = FeedForwardNetwork(4096,hidden , nbOfTopics, layers-1, rate)
        self.hidden = hidden
        self.rate = rate


    def learnNetwork(self, nbIter):
        training_names, training_distributions = self.load_dist("train")
        val_names, val_distributions = self.load_dist("val")
        training_feat_dict = self.dataprovider.getImageDict('train')
        val_feat_dict = self.dataprovider.getImageDict('val')
        test_names = self.dataprovider.getImageDict('test').keys()
        test_feats = np.array(self.dataprovider.getImageDict('test').values())

        val_feats = self.createFeatureMatrix(val_names, val_feat_dict)
        train_feats = self.createFeatureMatrix(training_names, training_feat_dict)

        self.layers = [Layer("Sigmoid", name = 'hidden',units=self.hidden), Layer("Softmax", name = 'out')]
        self.network = Regressor(self.layers, learning_rate = self.rate, n_iter = nbIter, valid_set = (val_feats, np.array(val_distributions)), verbose = True)

        self.network.fit(train_feats, np.array(training_distributions))

        val_end = self.network.predict(val_feats)
        test_end = self.network.predict(test_feats)
        self.save_split_values(test_names, test_end, 'test')
        self.save_split_values(val_names, val_end, 'val')
        print test_end
        #TODO hier save_split_values oproepen voor allebei


    def load_dist(self, split):
        filename = 'lda_images/models/image_topic_distribution_' + self.dataset + 'top' + str(
            self.nbOfTopics) + '_' + split + '.txt'
        return self.create_dist_dict(filename)

    '''
    Given a dictionary name -> feature and a matrix with names, return a matrix with the features in the same order
    as the given names.
    '''
    def createFeatureMatrix(self, names, featuredict):
        feat_mat = []
        for name in names:
            feat_mat.append(featuredict[name])
        return np.array(feat_mat)

    def create_dist_dict(self, filename):
        # dict = {}
        imagenames = []
        distributions = []
        f = open(filename)
        rawDist = []
        line = f.readline()
        while(line != ''):
            # print 'LINE', line
            split = line.split()
            if '[' in split and len(rawDist)!= 0:
                img, distribution = self.preprocess(rawDist)
                imagenames.append(img)
                distributions.append(distribution)
                rawDist = split
            else:
                rawDist.extend(split)
            line = f.readline()
        # Add the final image to the matrices
        img, distribution = self.preprocess(rawDist)
        imagenames.append(img)
        distributions.append(distribution)
        return imagenames, distributions

    def preprocess(self, rawDistribution):
        imgname = rawDistribution[0]
        distribution = []
        for i in range(2,len(rawDistribution)):
            modifiedNumber = str(rawDistribution[i]).replace(']', '')
            if modifiedNumber!= '':
                m = float(modifiedNumber)
                distribution.extend([m])
        return imgname, distribution

    '''
    Given a list of names, a list of numbers and a string representing the split, write the values to disk
    in an orderly manner
    '''
    def save_split_values(self, names, values, split):
        file = open('lda_images/models/image_topic_distribution_'+self.dataset+'_top'
                        +str(self.nbOfTopics)+'_'+split+'.txt', 'w')
        for i in range(len(names)):
            np.set_printoptions(suppress=True)
            prediction = values[i]
            img = names[i]
            file.write(img + ' ' + str(prediction) + '\n')
