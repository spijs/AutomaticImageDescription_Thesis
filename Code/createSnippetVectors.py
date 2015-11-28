import os
import scipy
import numpy as np
from nltk.stem.porter import *


''' stems a word by using the porter algorithm'''
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

'''
Creates a vocabulary, being a list of words
'''
def createVocabulary():
    dict = {}
    result = {}
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
            f= open('./Flickr30kEntities/sentence_snippets/'+filename)
            line = f.readline()
            while not (line == ""):
                for word in line.split():
                    word = stem(word.decode('utf-8'))
                    if(not word in dict):
                        dict[word]=1
                    else:
                        dict[word]+=1
                line = f.readline()
        for word in dict:
            if(dict[word] >= 5):
                result[word]=dict[word]
    return result.keys()

'''
Reads a set of documents, returns a dictionary containing the filename of the corresponding picture, and the
unweighted bag of words representation of the sentences in the documents, based on the given vocabulary
'''
def createOccurrenceVectors(vocabulary):

    return 0

if __name__ == "__main__":
    voc = createVocabulary()
    print len(voc)
    # occurrenceVectors = createOccurrenceVectors(voc)
