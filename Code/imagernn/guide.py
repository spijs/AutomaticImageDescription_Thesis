__author__ = 'Wout&Thijs'

import pickle
import os

def get_guide_size(guide_type):
    if guide_type=="image":
        return 4096

def get_guide(guide_type,im,L=None):
    if guide_type=="image":
        return get_image_guide(im)
    if guide_type=="snippetcca":
	    return get_cca_projection(im)
    if guide_type=="lda":
        return L

def get_image_guide(im):
    return im

def get_cca_projection(im):
    ccamodel = pickle.load(open("stackedCCAModel.p"))
    return ccamodel.getTransformedVector(im)



