__author__ = 'Thijs & Wout'

import os

from PIL import Image

def isLargeEnough(filename):
    '''
Given a filename, checks if the image behind that filename is bigger than 64x64
'''
    file = filename
    #print file
    try:
        image = Image.open("Flickr30kEntities/image_snippets/"+file)
    except IOError:
        # image not found. Is ok, many snippets dont have a corresponding image
        return False
    width, height = image.size
    return (width >=64 ) and (height >= 64)

''' This script filters images that are large enough and saves their filename to a file '''
if __name__ == "__main__":
    f = open("Flickr30kEntities/image_snippets/images.txt",'w+')
    g = open("Flickr30kEntities/images_snippets/images2.txt", 'w+')
    i=0
    for dirname, dirnames, filenames in os.walk('Flickr30kEntities/image_snippets'):
        for filename in filenames:
            if isLargeEnough(filename):
                if i%2==0:
                    f.write(filename+'\n')
                else:
                    g.write(filename+'\n')
            i += 1
