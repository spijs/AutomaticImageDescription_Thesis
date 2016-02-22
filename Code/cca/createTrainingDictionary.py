__author__ = 'Wout & thijs'

import os
import argparse
import numpy as np
from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
import scipy.io
from scipy import spatial
import pickle
from PIL import Image
from Code.imagernn.data_provider import getDataProvider


def main(params):
    dataset = params['dataset']
    dataprovider = getDataProvider(dataset)
    img_sentence_pair_generator = dataprovider.iterImageSentencePair()
    dict = {}
    result = {}
    stopwords = getStopwords()
    for pair in img_sentence_pair_generator:
        sentence = remove_common_words(pair['sentence']['tokens'],stopwords)
        for word in sentence:
            word = stem(word.decode('utf-8')).lower()
            if (not word in stopwords):
                if(not word in dict):
                    dict[word]=1
                else:
                    dict[word]+=1
    for word in dict:
            if(dict[word] >= 5):
                result[word]=dict[word]
    f = open("training_dictionary.txt", "w+")
    for w in result.keys():
        f.writelines(w+'\n')
    print('finished')

'''Returns a list containing the most frequent english words'''
def getStopwords():
        stopwords = set()
        file=open('../lda_images/english')
        for line in file.readlines():
            stopwords.add(line[:-1])
        return stopwords

''' stems a word by using the porter algorithm'''
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

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

''' Parses the given arguments'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')


    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)