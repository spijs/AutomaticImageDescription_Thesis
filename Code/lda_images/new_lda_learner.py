__author__ = 'Wout & thijs'

from imagernn.data_provider import getDataProvider
from lda_images.ff_nn import *
import numpy as np
from sknn.mlp import *

class LDANetworkLearner:


    def __init__(self, dataset, nbOfTopics, rate,hidden,layers, pert):
        '''
        :param dataset: dataset to use
        :param nbOfTopics: number of topics to use
        :param rate: learning rate
        :param hidden: number of hidden neurons
        :param layers: number of hidden layers. Currently not used since one hidden layer is hardcoded
        :param pert: whether or not to use the perturbed dataset
        '''
        self.nbOfTopics=nbOfTopics
        self.dataset = dataset
        self.dataprovider = getDataProvider(dataset, pert)
        self.pert = pert

        self.hidden = hidden
        self.rate = rate


    def learnNetwork(self, nbIter):
        '''
        Based on the given number of iterations, train a neural network to predict topic distributions and predict
        the values for the test and validation set
        :param nbIter: number of iterations to train for
        '''
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


    def load_dist(self, split):
        '''
        Given a split, load the corresponding image topic distribution
        :param split: test/train/val
        :return: the image-topic distributions corresponding to the given split
        '''
        pert_str = ''
        if self.pert:
            pert_str ='_pert_'
        filename = 'lda_images/models/image_topic_distribution_' + self.dataset + 'top' + str(
            self.nbOfTopics) + '_' + split + pert_str +  '.txt'
        return self.create_dist_dict(filename)


    def createFeatureMatrix(self, names, featuredict):
        '''
        Given a dictionary mapping image names to image features and a list with names,
        return a matrix with the image features in the same order
        as the given names.
        :param names: list of image names
        :param featuredict: dictionary containing names->features mapping
        :return: matrix with the features ordered like the names in the input list
        '''
        feat_mat = []
        for name in names:
            feat_mat.append(featuredict[name])
        return np.array(feat_mat)

    def create_dist_dict(self, filename):
        '''
        :param filename: file to read
        :return: a list of image names and topic distributions ordered in the same way
        '''
        imagenames = []
        distributions = []
        f = open(filename)
        rawDist = []
        line = f.readline()
        while(line != ''):
            split = line.split()
            if '[' in split and len(rawDist)!= 0:
                img, distribution = self.preprocess(rawDist)
                imagenames.append(img)
                distributions.append(distribution)
                rawDist = split
            else:
                rawDist.extend(split)
            line = f.readline()
        img, distribution = self.preprocess(rawDist)
        imagenames.append(img)
        distributions.append(distribution)
        return imagenames, distributions

    def preprocess(self, rawDistribution):
        '''
        Given a list, containing both image name and topic distribution, extract the name and distribution and return
        :param rawDistribution: list to process
        :return: extracted image name and distribution
        '''
        imgname = rawDistribution[0]
        distribution = []
        for i in range(2,len(rawDistribution)):
            modifiedNumber = str(rawDistribution[i]).replace(']', '')
            if modifiedNumber!= '':
                m = float(modifiedNumber)
                distribution.extend([m])
        return imgname, distribution


    def save_split_values(self, names, values, split):
        '''
        Given a list of names, a list of numbers and a string representing the split, write the values to disk
        in an orderly manner
        :param names: list of image names
        :param values: list of topic distributions
        :param split: test/train/val
        '''
        pert_str = ''
        if self.pert:
            pert_str ='_pert'
        file = open('lda_images/models/image_topic_distribution_'+self.dataset+'_top'
                        +str(self.nbOfTopics)+'_'+split+ pert_str + '.txt', 'w')
        for i in range(len(names)):
            np.set_printoptions(suppress=True)
            prediction = values[i]
            img = names[i]
            file.write(img + ' ' + str(prediction) + '\n')
