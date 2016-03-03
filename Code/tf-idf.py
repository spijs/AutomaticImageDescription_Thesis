__author__ = 'Wout & thijs'

import os
import sys
import argparse
import numpy as np
from nltk.stem.porter import *
# from sklearn.cross_decomposition import CCA
sys.path.append("..")
import rcca
import scipy.io
from scipy import spatial
import pickle
from PIL import Image
from imagernn.data_provider import getDataProvider
import sys
import operator

def getOccurenceVectorsAndImages():
    vocabulary = readVocabulary("cca/training_dictionary.txt")
    idf = {}
    dataprovider = getDataProvider('flickr30k')
    pairGenerator = dataprovider.iterImageSentencePair()
    stopwords = getStopwords()
    current = 0
    for pair in pairGenerator:
        current+= 1
        if current % 1000 ==0:
            print "Processing pair " + str(current)
        sentence = remove_common_words(pair['sentence']['tokens'],stopwords)
        wordcount = 0
        done = []
        for word in sentence:
            stemmed = stem(word.decode('utf-8')).lower()
            if stemmed in vocabulary and stemmed not in done:
                idf[stemmed] += 1
                done.append(stemmed)
    idf = {k: np.log(current/v) for k, v in idf.iteritems()}
    sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
    for k, v in sorted_idf.iteritems():
        print "Word: " + k + " IDF: " + str(v)
    output = open('idf.p', 'wb')
    pickle.dump(idf, output)
    output.close()

''' stems a word by using the porter algorithm'''
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

'''
Creates a vocabulary based on a folder. Returns a list of words
'''
def readVocabulary(filename):
    result = []
    voc = open(filename)
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result

'''
Given a sentence, return a copy of that sentence, stripped of words that are in the provided stopwords
'''
def remove_common_words(sentence,stopwords):
        #s = set(stopwords.words('english'))
        stopwords.add(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word.lower() in stopwords and len(word)>2:
                result.append(word.lower())
        return result

'''Returns a list containing the most frequent english words'''
def getStopwords():
    stopwords = set()
    file=open('lda_images/english')
    for line in file.readlines():
        stopwords.add(line[:-1])
    return stopwords