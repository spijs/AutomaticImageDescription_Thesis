__author__ = 'Wout & thijs'

import os
import sys
import argparse
import numpy as np
from nltk.stem.porter import *
sys.path.append("..")
import rcca
import pickle
from imagernn.data_provider import getDataProvider

def preprocess():
    '''
    generate the image and sentence matrices of the given dataset and write it to disk
    :return:
    '''
    dataset = "flickr30k" # hardcoded
    os.chdir("..")
    dataprovider = getDataProvider(dataset, pert=1)
    os.chdir("cca")
    img_sentence_pair_generator = dataprovider.iterImageSentencePair()
    print "Reading Vocabulary..."
    vocabulary = readVocabulary("training_dictionary_pert.txt")
    print "Done"
    print "Creating sentence vectors..."
    occurrences, idf, images = getOccurenceVectorsAndImages(vocabulary, img_sentence_pair_generator)
    print "Done"
    print "Weighing vectors"
    weightedVectors = weight_tfidf(occurrences, idf)
    pair = image_sentence_matrix_pair(images, weightedVectors)
    pair_file = open("imagesentencematrix_pert.p", 'wb')
    pickle.dump(pair, pair_file)
    pair_file.close()

def create_mat_files():
    '''
    Load a image and sentence matrix into memory and write them to disk in two separate files.
    :return:
    '''
    print "Loading data into memory"
    load_from = open("imagesentencematrix_pert.p", 'rb') # filename is hardcoded
    matrixpair = pickle.load(load_from)
    load_from.close()
    images = np.array(matrixpair.images)
    print "Image dimensions " + str(images.shape)
    sentences = np.array(matrixpair.sentences)
    print "Sentence dimensions " + str(sentences.shape)
    np.savetxt("sentences_pert.txt", sentences)
    np.savetxt("images_pert.txt", images)

class image_sentence_matrix_pair:
    '''
    Class containing an image matrix and a sentence matrix
    '''
    def __init__(self, images, sentences):
        self.images = images
        self.sentences = sentences


def main(params):
    '''
    Load an image and sentence matrix into memory, learn a CCA model on them and write this model to disk
    :param params
    '''
    print "Loading data into memory"
    load_from = open("imagesentencematrix.p", 'rb')
    matrixpair = pickle.load(load_from)
    load_from.close()
    images = np.array(matrixpair.images)
    print "Image dimensions " + str(images.shape)
    sentences = np.array(matrixpair.sentences)
    print "Sentence dimensions " + str(sentences.shape)
    print "Done"
    print "Learning CCA"
    cca = rcca.CCA(kernelcca=False, numCC=256, reg=0.)
    cca.train([images, sentences])
    print "writing results to pickle"
    pickle_dump_file = open("../data/trainingCCA.p",'w+')
    pickle.dump(cca, pickle_dump_file)
    pickle_dump_file.close()
    print('finished')

def getStopwords():
    '''
    :return: a list containing the most frequent english words
    '''
    stopwords = set()
    file=open('../lda_images/english')
    for line in file.readlines():
        stopwords.add(line[:-1])
    return stopwords

def stem(word):
    '''
    :param word:
    :return: the word, stemmed by the porter algorithm
    '''
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def weight_tfidf(documents, inv_freq):
    result = []
    for i in range(len(documents)):
        doc = documents[i]
        result.append(doc * inv_freq)
    return result

def remove_common_words(sentence,stopwords):
    '''
    :param sentence:
    :param stopwords:
    :return: the given sentence, stripped of stopwords and words with length <3
    '''
    stopwords.add(' ') #add spaces to stopwords
    result = []
    for word in sentence:
        if not word.lower() in stopwords and len(word)>2:
            result.append(word.lower())
    return result

def getOccurenceVectorsAndImages(vocabulary, pairGenerator):
    '''

    :param vocabulary
    :param pairGenerator: generator of image sentence pairs
    :return: term frequency vectors for each sentence, the idf weight for each word and a matrix containing all image
    representations
    '''
    stopwords = getStopwords()
    images = []
    idf = np.zeros(len(vocabulary))
    current = 0
    result = []
    for pair in pairGenerator:
        current+= 1
        if current % 1000 ==0:
            print "Processing pair " + str(current)
        sentence = remove_common_words(pair['sentence']['tokens'],stopwords)
        wordcount = 0
        row = np.zeros(len(vocabulary))
        for word in sentence:
            stemmed = stem(word.decode('utf-8')).lower()
            if stemmed in vocabulary:
                wordcount += 1
                i = vocabulary.index(stemmed.lower())
                row[i] += 1
            if wordcount:
                row = row / wordcount
            for w in range(len(row)):
                if row[w] > 0:
                    idf[w] += 1
        result.append(row)
        images.append(pair['image']['feat'])
    idf = len(result) / idf
    return result, idf, images

def readVocabulary(filename):
    '''
    :param filename: file to read from
    :return: the vocabulary stored in the given file
    '''
    result = []
    voc = open(filename)
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')


    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)
