__author__ = 'Wout'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np

if __name__ == "__main__":
    dataprovider = getDataProvider('flickr30k', 120)

    dataprovider.load_topic_models('flickr30k', 120)

    dataprovider.testfunction('val')

    top = dataprovider.topics
    print "amount of topics", len(top)

