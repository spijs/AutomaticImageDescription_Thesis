__author__ = 'Wout'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np

if __name__ == "__main__":
    dataprovider = getDataProvider('flickr30k')
    print dataprovider
    dataprovider.load_topic_models('flickr30k', 120)
    print 'Hier'
    batches = dataprovider.iterImageSentencePairBatch(split = 'val')
    for batch in batches:
        print batch
