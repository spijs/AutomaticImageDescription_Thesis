__author__ = 'Wout & thijs'

from data_provider import getDataProvider
import math
import os

def main():
    dataset = 'flickr30k'
    os.chdir("..")
    dataprovider = getDataProvider(dataset)
    os.chdir("imagernn")
    img_sentence_pair_generator = dataprovider.iterImageSentencePair()
    mean = 0.0
    nb_of_sentences = 0.0

    for pair in img_sentence_pair_generator:
        l = len(pair['sentence']['tokens'])
        mean = mean+l
        nb_of_sentences=nb_of_sentences+1
    mean = mean/nb_of_sentences

    img_sentence_pair_generator = dataprovider.iterImageSentencePair()
    dev = 0.0
    for pair in img_sentence_pair_generator:
        l = len(pair['sentence']['tokens'])
        d = math.pow(mean-l,2)
        dev = dev+d
    dev = math.sqrt(dev/nb_of_sentences)
    print('mean: ', mean)
    print('std.dev: ',dev)
