__author__ = 'Wout'

from imagernn.data_provider import getDataProvider
from lda_images.ff_nn import *

class LDANetworkLearner:


    def __init__(self, dataset, nbOfTopics, rate):
        self.nbOfTopics=nbOfTopics
        self.dataset = dataset
        self.dataprovider = getDataProvider(dataset)
        self.network = FeedForwardNetwork(4096, 256, nbOfTopics, rate)


    # Train a simple FF neural network based on the topic distributions that were calculated earlier
    # First creates a dictionary to map the image names onto the topic distributions,
    # then samples random images from the dataprovider, and perform a forward and backward step
    def learnNetwork(self, iterations):
        filename = 'lda_images/image_topic_distribution_' + self.dataset + 'top' + str(self.nbOfTopics) + '.txt'
        self.dictionary = self.create_dist_dict(filename)
        image_sentence_pair_generator = self.dataprovider.iterImageSentencePair(split = 'train')

        for i in range(iterations):
            pair = self.dataprovider.sampleImageSentencePair()
            # if(pair['image']['filename']!=last_img):
            features = pair['image']['feat']
            dist = self.dictionary[pair['image']['filename']]
            # print 'FEATURES', len(features)
            # print 'DIST', len(dist)
            # print 'DIST', dist
            self.network.forward(features)
            self.network.backward(dist)


        filename = 'networkweights_'+self.dataset +'_' + str(self.nbOfTopics) + '.txt'
        self.network.writeResults(filename)




    def create_dist_dict(self, filename):
        dict = {}
        f = open(filename)
        rawDist = []
        line = f.readline()
        while(line != ''):
            # print 'LINE', line
            split = line.split()
            if '[' in split and len(rawDist)!= 0:
                img, distribution = self.preprocess(rawDist)
                dict[img] = distribution
                rawDist = split
            else:
                rawDist.extend(split)
            line = f.readline()
        img, distribution = self.preprocess(rawDist)
        dict[img] = distribution
        return dict

    def preprocess(self, rawDistribution):
        # print 'RAAAWWWWW', rawDistribution
        imgname = rawDistribution[0]
        distribution = []
        for i in range(2,len(rawDistribution)):
            modifiedNumber = str(rawDistribution[i]).replace(']', '')
            # print modifiedNumber
            if modifiedNumber!= '':
                m = float(modifiedNumber)
                distribution.extend([m])
        if len(distribution) != 120:
            print 'LENGTH', len(distribution)
        return imgname, distribution
    def learnOneStep(self, image, distribution):
        self.network.forward(image)
        self.network.backward(distribution)
