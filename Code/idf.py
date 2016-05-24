__author__ = 'Wout & thijs'

import sys
import numpy as np
from nltk.stem.porter import *
sys.path.append("..")
import pickle
from imagernn.data_provider import getDataProvider
import operator

def print_and_save_previously_generated():
    '''
    sorts dictionary of words with idf scores and saves it as pickle file.
    '''
    input = open('idf.p','rb')
    idf = pickle.load(input)
    sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
    output = open('sorted_idf.txt','w')
    for pair in sorted_idf:
        print "Word: " + pair[0] + " IDF: " + str(pair[1])
        output.write(pair[0] + ": " + str(pair[1])+"\n")
    output.close()

def getOccurenceVectorsAndImages():
    '''
    Creates a dictionary of stemmed words and idf scores for the test set, saves this as a pickle file.
    NOTE: All files and datasets are still hardcoded!
    '''
    vocabulary = readVocabulary("cca/training_dictionary.txt")
    idf = {}
    for word in vocabulary:
        idf[word] = 0
    dataprovider = getDataProvider('flickr30k')
    pairGenerator = dataprovider.iterImageSentencePair()
    stopwords = getStopwords()
    current = 0
    for pair in pairGenerator:
        current+= 1
        if current % 1000 ==0:
            print "Processing pair " + str(current)
        sentence = remove_common_words(pair['sentence']['tokens'],stopwords)
        done = []
        for word in sentence:
            stemmed = stem(word.decode('utf-8')).lower()
            if stemmed in vocabulary and stemmed not in done:
                idf[stemmed] += 1
                done.append(stemmed)
    idf = {k: np.log(current/v) for k, v in idf.iteritems()}
    sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
    for pair in sorted_idf:
        print "Word: " + pair[0] + " IDF: " + str(pair[1])
    output = open('idf.p', 'wb')
    pickle.dump(idf, output)
    output.close()

def stem(word):
    ''' stems a word by using the porter algorithm'''
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def readVocabulary(filename):
    '''
    Creates a vocabulary based on a folder. Returns a list of words
    '''
    result = []
    voc = open(filename)
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result

def remove_common_words(sentence,stopwords):
    '''
    Given a sentence, return a copy of that sentence, stripped of words that are in the provided stopwords
    '''
    #s = set(stopwords.words('english'))
    stopwords.add(' ') #add spaces to stopwords
    result = []
    for word in sentence:
        if not word.lower() in stopwords and len(word)>2:
            result.append(word.lower())
    return result

def getStopwords():
    '''Returns a list containing the most frequent english words'''
    stopwords = set()
    file=open('lda_images/english')
    for line in file.readlines():
        stopwords.add(line[:-1])
    return stopwords