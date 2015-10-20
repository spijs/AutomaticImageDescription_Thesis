import os
import random
import scipy.io
import codecs
import numpy
from PIL import Image

from collections import defaultdict

# Om te gebruiker: zorg dat eerst getAnnotations.m is gerund
# Resultaat: snippets worden uit de afbeeldingen gehaald, en opgeslagen als <imageID>_<annotationID>_<annotationNb>.jpg
# annotationNb wordt gebruikt als een snippet meerdere keren voorkomt (in vercshillende zinnen)
# 
# 
# 

for dirname, dirnames, filenames in os.walk('./matFiles'):
    # # print path to all subdirectories first.
    for filename in filenames:
    	fileToLoad = os.path.join('matFiles',filename[0:len(filename)])
    	imgpath = 'G:/flickr30k-images/flickr30k-images/' + filename[0:(len(filename)-4)] + '.jpg'
    	img = Image.open(imgpath)
        annotations = scipy.io.loadmat(fileToLoad)
        annotation_struct = annotations['annotations'][0][0]
        imgname = annotation_struct[0][0]
        # print imgname
        # # print img
        boxIDs = annotation_struct[3]
        dimensions = annotation_struct[4]
        mapping = annotation_struct[5]
        boxid_extracted = []
        for i in range(len(boxIDs)):
            newid = boxIDs[i][0][0]
            boxid_extracted.append(newid)

        for i in range(len(boxid_extracted)):
            curr_map = mapping[i][0]
            curr_map = [item for sublist in curr_map for item in sublist]
            # print 'map', curr_map, 'length', len(curr_map)
            if len(curr_map) > 0:
                # print 'length mapping', len(curr_map), 'mapp', curr_map
                # print 'dimensions', dimensions
                for j in range(len(curr_map)):
                    # print 'J', j
                    # print curr_map[j]
                    label = dimensions[curr_map[j]-1][0][0]
                    # print 'LABEL', label
                    while len(label) ==1:
                      label = label[0]
                    if len(label) != 0:
                        box = (label[0], label[1], label[2], label[3])
                        region = img.crop(box)
                        regionsave = 'image_snippets/' + str(imgname) + '_' + boxid_extracted[i] +'_'+str(j+1) + '.jpg'
                        # # print 'regionsave', regionsave 
                        region.save(regionsave)
            
        # # print dimensions[0][0][0][0]
        # # print boxid_extracted[0]
