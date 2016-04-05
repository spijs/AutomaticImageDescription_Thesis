__author__ = 'Wout & thijs'

import json
import os
import random
import scipy.io
import numpy as np
import codecs
from collections import defaultdict

class BasicDataProvider:
    def __init__(self, dataset, pert):
        print 'Initializing data provider for dataset %s...' % (dataset, )
        self.topics = None
        self.pert= pert
        # !assumptions on folder structure
        self.dataset_root = os.path.join('data', dataset)
        self.image_root = os.path.join('data', dataset, 'imgs')

        # load the dataset into memory
        if not pert:
            dataset_path = os.path.join(self.dataset_root, 'dataset.json')
        else:
            dataset_path = os.path.join(self.dataset_root, 'pert_dataset.json')
        print 'BasicDataProvider: reading %s' % (dataset_path, )
        self.dataset = json.load(open(dataset_path, 'r'))

        # load the image features into memory
        features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
        print 'BasicDataProvider: reading %s' % (features_path, )
        features_struct = scipy.io.loadmat(features_path)
        self.features = features_struct['feats']

        # group images by their train/val/test split into a dictionary -> list structure
        self.split = defaultdict(list)
        for img in self.dataset['images']:
            self.split[img['split']].append(img)

    # "PRIVATE" FUNCTIONS
    # in future we may want to create copies here so that we don't touch the
    # data provider class data, but for now lets do the simple thing and
    # just return raw internal img sent structs. This also has the advantage
    # that the driver could store various useful caching stuff in these structs
    # and they will be returned in the future with the cache present
    def _getImage(self, img):
        """ create an image structure for the driver """

        # lazily fill in some attributes
        if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
        if not 'feat' in img: # also fill in the features
            feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
            img['feat'] = self.features[:,feature_index]
        if self.topics:
            img['topics'] = self.topics[img['filename']]
        return img

    '''
    Returns a dictionary mapping image names to  image features
    '''
    def getImageDict(self, split):
        imagedict = {}
        for i,img in enumerate(self.split[split]):
            imagedict[img['filename']] = self._getImage(img)['feat']
        return imagedict

    def _getSentence(self, sent):
        """ create a sentence structure for the driver """
        # NOOP for now
        return sent

    def load_topic_models(self,dataset,topics):
         # load the topic distributions into memory
        pert_string = train_string = ''
        if self.pert:
            pert_string = '_pert'
            train_string = '_'
        topic_root = os.path.join('lda_images/models/image_topic_distribution_'+dataset+'top')
        self.topics = {}
        train_file = topic_root+str(topics)+'_train'+pert_string+train_string+'.txt'
        self.topics = self.create_dist_dict(train_file, self.topics)
        for split in ['test', 'val']:
            f = os.path.join('lda_images/models/image_topic_distribution_'+dataset+'_top'+str(topics)+'_'+split+pert_string+'.txt')
            self.topics = self.create_dist_dict(f, self.topics)
        print 'amount of topics', len(self.topics)

    def load_cca_weights(self,nb_components, pert = None):
        pert_str = ''
        if pert:
            pert_str = '_pert'
        return np.loadtxt('cca/imageprojection_'+str(nb_components)+pert_str+'.txt', delimiter = ',')

    def create_dist_dict(self, filename, dict):
        if os.path.isfile(filename):
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
        else:
            print 'file not found'
        return dict

    # PUBLIC FUNCTIONS

    def getSplitSize(self, split, ofwhat = 'sentences'):
        """ return size of a split, either number of sentences or number of images """
        if ofwhat == 'sentences':
            return sum(len(img['sentences']) for img in self.split[split])
        else: # assume images
            return len(self.split[split])

    def sampleImageSentencePair(self, split = 'train'):
        """ sample image sentence pair from a split """
        images = self.split[split]

        img = random.choice(images)
        sent = random.choice(img['sentences'])

        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        if self.topics:
            #print 'setting topic dist', self.topics[img['filename']]
            out['topics'] = self.topics[img['filename']]

        return out

    def iterImageSentencePair(self, split = 'train', max_images = -1):
        for i,img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images: break
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = self._getSentence(sent)
                yield out

    def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
        batch = []
        for i,img in enumerate(self.split[split]):
            if max_images >= 0 and i >= max_images: break
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = self._getSentence(sent)
                if self.topics:
                    # print 'setting topic dist', self.topics[img['filename']]
                    out['topics'] = self.topics[img['filename']]
                batch.append(out)
                if len(batch) >= max_batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def iterSentences(self, split = 'train'):
        for img in self.split[split]:
            for sent in img['sentences']:
                yield self._getSentence(sent)

    def iterImages(self, split = 'train', shuffle = False, max_images = -1):
        imglist = self.split[split]
        ix = range(len(imglist))
        if shuffle:
            random.shuffle(ix)
        if max_images > 0:
            ix = ix[:min(len(ix),max_images)] # crop the list
        for i in ix:
            yield self._getImage(imglist[i])

    def testfunction(self, split):
        for i,img in enumerate(self.split[split]):
            for sent in img['sentences']:
                out = {}
                out['image'] = self._getImage(img)
                out['sentence'] = self._getSentence(sent)
                if self.topics:
                    #print 'setting topic dist', self.topics[img['filename']]
                    out['topics'] = self.topics[img['filename']]

    def preprocess(self, rawDistribution):
        imgname = rawDistribution[0]
        distribution = []
        for i in range(2,len(rawDistribution)):
            modifiedNumber = str(rawDistribution[i]).replace(']', '')
            # print modifiedNumber
            if modifiedNumber and not ' ' in modifiedNumber:
                m = float(modifiedNumber)
                distribution.extend([m])
        return imgname, distribution

    def getTopic(self, imgName):
        return self.topics[imgName]
def getDataProvider(dataset,pert=None):
    """ we could intercept a special dataset and return different data providers """
    assert dataset in ['flickr8k', 'flickr30k', 'coco'], 'dataset %s unknown' % (dataset, )
    return BasicDataProvider(dataset,pert)



