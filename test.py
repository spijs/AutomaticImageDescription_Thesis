import json
import os
import random
import scipy.io
import codecs
import numpy
import matlab.engine
from collections import defaultdict

eng = matlab.engine.start_matlab()
eng.triarea(nargout=0)

struct_read = scipy.io.loadmat('Flickr30kEntities/ann.mat')

annotation_struct = struct_read['ann1'][0][0]

img = annotation_struct[0][0]
boxIDs = annotation_struct[3]
dimensions = annotation_struct[4]
mapping = annotation_struct[5]

print img
print len(boxIDs)
boxid_extracted = []
for i in range(len(boxIDs)):
	newid = boxIDs[i][0][0]
	boxid_extracted.append(newid)
print dimensions
print mapping[0][0]