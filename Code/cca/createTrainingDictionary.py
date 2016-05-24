__author__ = 'Wout & thijs'

import os
import argparse
from nltk.stem.porter import *
import sys
sys.path.append("../imagernn")

from data_provider import getDataProvider


def main(params):
    '''
    iterate over all image sentence pairs in the dataset, and write a dictionary containing all words used more
    than 5 times
    :param params
    '''
    dataset = params['dataset']
    os.chdir("..")
    dataprovider = getDataProvider(dataset, pert = 1)
    os.chdir("cca")
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
    f = open("training_dictionary_pert.txt", "w+")
    for w in result.keys():
        f.writelines(w+'\n')
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
    :param word
    :return: the given word, stemmed using the porter algorithm
    '''
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def remove_common_words(sentence,stopwords):
    '''
    :param sentence
    :param stopwords
    :return: copy of the given sentence stripped of stopwords and words with length <3
    '''
    stopwords.add(' ') #add spaces to stopwords
    result = []
    for word in sentence:
        if not word.lower() in stopwords and len(word)>2:
            result.append(word.lower())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')


    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)