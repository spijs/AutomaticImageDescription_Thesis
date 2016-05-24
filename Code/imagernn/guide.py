__author__ = 'Wout&Thijs'

import pickle
import numpy as np

def get_guide_size(guide_type,lda=None):
    '''
    :return: Returns the size of the guide corresponding to the given guide_type
    '''
    if guide_type=="image":
        return 4096
    if guide_type=="lda":
       return lda
    if guide_type=="cca":
        return lda.shape[1]

def get_guide(guide_type,im,L=None):
    '''
    Returns the guide vector corresponding to the given image and guide type.
    '''
    if guide_type=="image":
        return get_image_guide(im)
    if guide_type=="snippetcca":
	    return get_cca_projection(im)
    if guide_type=="cca":
        return get_image_projection(im, L)
    if guide_type=="lda":
        return L

def get_image_guide(im):
    '''
    Simply returns given image.
    '''
    return im

def get_cca_projection(im):
    '''
    Returns the precomputed image extended CCA projection of the given image.
    NOT USED
    '''
    ccamodel = pickle.load(open("stackedCCAModel.p"))
    return ccamodel.getTransformedVector(im)

def get_image_projection(image, weights):
    '''
    Returns the CCA projection of the image using the given weights.
    '''
    return np.dot(image, weights)



