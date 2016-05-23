__author__ = "Wout & Thijs"

import os
from nltk.stem.porter import *
from PIL import Image


def isLargeEnough(filename):
    '''
    :param filename:
    :return: True if the image behind that filename is bigger than 64x64
    '''
    file = filename+".jpg"
    try:
        image = Image.open("../Flickr30kEntities/image_snippets/"+file)
    except IOError:
        return False
    width, height = image.size
    return (width >= 64) and (height >= 64)

def stem(word):
    '''
    :param word
    :return: word stemmed by the porter algorithm
    '''
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def getStopwords():
    '''
    :return: a list containing the most frequent English words
    '''
    stopwords = set()
    file=open('../lda_images/english')
    for line in file.readlines():
        stopwords.add(line[:-1])
    return stopwords


def main():
    '''
    Based on the files in an hardcoded folder, create a dictionary containing stemmed words and write it to disk
    '''
    dict = {}
    result = {}
    stopwords = getStopwords()
    current = 0
    for dirname, dirnames, filenames in os.walk('../Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
            current += 1
            if current % 1000 == 0:
                print "Preprocessing sentence: " + str(current)
            f = open('../Flickr30kEntities/sentence_snippets/' + filename)
            line = f.readline()
            sentenceid = 1
            # print filename
            while not (line == ""):
                if isLargeEnough(filename[0:-4] + '_' + str(sentenceid)):
                    for word in line.split():
                        word = stem(word.decode('utf-8')).lower()
                        if (not word in stopwords):
                            if (not word in dict):
                                dict[word] = 1
                            else:
                                dict[word] += 1
                line = f.readline()
                sentenceid += 1
        for word in dict:
            if (dict[word] >= 5):
                result[word] = dict[word]
    words = result.keys()
    f = open("complete_dictionary.txt", 'w+')
    for w in words:
        f.writelines(w + '\n')
    f.close()


if __name__ == "__main__":
    main()
