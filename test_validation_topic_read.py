__author__ = 'Wout'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np

if __name__ == "__main__":
    dataprovider = getDataProvider('flickr30k')
    dataprovider.load_topic_models('flickr30k', 120)
    batches = dataprovider.iterImageSentencePairBatch(split = 'val')
    i=0
    for batch in batches:
        print i
        i+=1
        print (x['topics'] for x in batch)
