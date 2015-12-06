__author__ = 'wout'

import os
import numpy as np
from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
import scipy.io
from scipy import spatial
import pickle
from PIL import Image
from imagernn.data_provider import getDataProvider
from stackedCCAModel import *

'''
Given a filename, checks if the image behind that filename is bigger than 64x64
'''
def isLargeEnough(filename):
    file = filename+".jpg"
    #print file
    try:
        image = Image.open("Flickr30kEntities/image_snippets/"+file)
    except IOError:
        # image not found. Is ok, many snippets dont have a corresponding image
	    return False
    width, height = image.size
    return (width >=400 ) and (height >= 400)

if __name__ == "__main__":
    for dirname, dirnames, filenames in os.walk('Flickr30kEntities/image_snippets'):
        f = open("Flickr30kEntities/images_snippets/images.txt", 'w+')
        g = open("Flickr30kEntities/images_snippets/images2.txt", 'w+')
        i = 0
        for filename in filenames:
            if isLargeEnough(filename):
                if i%2==0:
                    f.write(filename+'\n')
                else:
                    g.write(filename+'\n')
            i += 1