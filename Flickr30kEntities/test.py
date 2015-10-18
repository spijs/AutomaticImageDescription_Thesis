import matlab.engine
import os
import random
import scipy.io
import codecs
import numpy

from collections import defaultdict

eng = matlab.engine.start_matlab()
for dirname, dirnames, filenames in os.walk('./tmp'):
    # print path to all subdirectories first.
    for filename in filenames:
        eng.getAnnotations(os.path.join(dirname, filename), nargout = 0)
        annotations = scipy.io.loadmat('annotations.mat')
        annotation_struct = annotations['annotations'][0][0]
        img = annotation_struct[0][0]
        print img
        boxIDs = annotation_struct[3]
        dimensions = annotation_struct[4]
        mapping = annotation_struct[5]
        print len(boxIDs)
        boxid_extracted = []
        for i in range(len(boxIDs)):
        	newid = boxIDs[i][0][0]
        	boxid_extracted.append(newid)
        print dimensions
# print mapping[0][0]
# eng = matlab.engine.start_matlab()
# print len(eng.getAnnotations('Annotations/36979.xml'))

# print struct_read


# annotation_struct = struct_read['ann1'][0][0]

# img = annotation_struct[0][0]
# boxIDs = annotation_struct[3]
# dimensions = annotation_struct[4]
# mapping = annotation_struct[5]

# print img
# print len(boxIDs)
# boxid_extracted = []
# for i in range(len(boxIDs)):
# 	newid = boxIDs[i][0][0]
# 	boxid_extracted.append(newid)
# print dimensions
# print mapping[0][0]