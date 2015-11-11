__author__ = 'Wout'

from imagernn.data_provider import getDataProvider
from lda_images.ff_nn import *
import numpy
import copy
import sys

class LDANetworkLearner:


    def __init__(self, dataset, nbOfTopics, rate,hidden,layers):
        self.nbOfTopics=nbOfTopics
        self.dataset = dataset
        self.dataprovider = getDataProvider(dataset)
        self.network = FeedForwardNetwork(4096,hidden , nbOfTopics, layers-1, rate)
        self.hidden = hidden
        self.layers = layers-1
        self.rate = rate

    # Train a simple FF neural network based on the topic distributions that were calculated earlier
    # First creates a dictionary to map the image names onto the topic distributions,
    # then samples random images from the dataprovider, and perform a forward and backward step
    def learnNetwork(self, iterations):
        self.dictionary = {}
        for split in ['train', 'test', 'val']:
            filename = 'lda_images/models/image_topic_distribution_' + self.dataset + 'top' + str(self.nbOfTopics) + '_'+split+'.txt'
            self.dictionary = self.create_dist_dict(filename, self.dictionary)
        # image_sentence_pair_generator = self.dataprovider.iterImageSentencePair(split = 'train')
        # validationset = self.dataprovider.iterImageSentencePair(split = 'val')
        # self.topicnetworks = []
        # for i in range(12):
        #     self.topicnetworks.extend([FeedForwardNetwork(4096, 256, 1, self.rate)])
        # # validationError = sys.maxint
        self.validationError = (sys.maxint-1)

        for i in range(iterations):
            if i % 100 == 0:
                print 'Iteration', i+1, 'of', iterations
            pair = self.dataprovider.sampleImageSentencePair()
            # if(pair['image']['filename']!=last_img):
            features = pair['image']['feat']
            dist = self.dictionary[pair['image']['filename']]
            # print 'FEATURES', len(features)
            # print 'DIST', len(dist)
            # print 'DIST', dist
            # for networkID in len(self.topicnetworks):
            self.network.forward(features)
            self.network.backward(dist)
            if i % 2000  == 1999 :
                last_img = ''
                intermediate_error = 0.0
                for j in range(1000):
                    validationPair = self.dataprovider.sampleImageSentencePair('val')
                    prediction = self.network.predict(validationPair['image']['feat'])
                    correct = self.dictionary[validationPair['image']['filename']]
                    err = -sum(correct*log(prediction))
                    intermediate_error += err
                print 'validation error', intermediate_error
                if i % 30000 == 29999:
                    if intermediate_error > self.validationError:
                        print intermediate_error
                        print 'No more improvement'
                        break
                    else:
                        self.bestNetwork = copy.deepcopy(self.network)
                        print 'Validation Error', intermediate_error
                        self.validationError = intermediate_error

        self.create_test_validation()

    def create_test_validation(self):
        for split in ['test', 'val']:
            set = self.dataprovider.iterImageSentencePair(split = split)
            file = open('lda_images/models/image_topic_distribution_'+self.dataset+'_top'
                        +str(self.nbOfTopics)+'_'+split+'_'+str(self.hidden)+'_' + str(self.layers)+'_' + str(self.rate)+ '_' +str(self.validationError)+'.txt', 'w')
            numpy.set_printoptions(suppress=True)
            for pair in set:
                prediction = self.bestNetwork.predict(pair['image']['feat'])
                img = pair['image']['filename']
                file.write(img + ' ' + str(prediction) + '\n')



    def testNetwork(self):
        topicnamelist = self.createTopicList()
        for i in range(10):
            testPair = self.dataprovider.sampleImageSentencePair('test')
            prediction = self.network.predict(testPair['image']['feat'])
            #print prediction
            #print type(prediction)
            print testPair['image']['filename']
            sortedpred = sorted(prediction)
            sortedpred = sortedpred[::-1]
            prediction = prediction.tolist()
            #print 'sorted', sortedpred
            print 'Best topics\n'
            for j in range(5):
                index = prediction.index(sortedpred[j])
                print ('topic:' + str(topicnamelist[index]) + ' probability'+ str(sortedpred[j]))


    def createTopicList(self):
        file = open('lda_images/models/topicnames'+str(self.nbOfTopics)+'.txt')
        list = []
        line = file.readline()
        while line != '':
            list.extend([line])
            line = file.readline()
        return list

    def create_dist_dict(self, filename, dict):
        # dict = {}
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
        imgname = rawDistribution[0]
        distribution = []
        for i in range(2,len(rawDistribution)):
            modifiedNumber = str(rawDistribution[i]).replace(']', '')
            # print modifiedNumber
            if modifiedNumber!= '':
                m = float(modifiedNumber)
                distribution.extend([m])
        return imgname, distribution
    def learnOneStep(self, image, distribution):
        self.network.forward(image)
        self.network.backward(distribution)
