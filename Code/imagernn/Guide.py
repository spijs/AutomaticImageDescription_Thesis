__author__ = 'Wout&Thijs'

def get_guide_size(guide_type):
    if guide_type=="image":
        return 256

def get_guide(guide_type,im):
    if guide_type=="image":
        return get_image_guide(im)

def get_image_guide(im):
    return im

