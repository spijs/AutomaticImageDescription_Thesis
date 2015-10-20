import os
import random
import scipy.io
import codecs
import numpy
from PIL import Image

from collections import defaultdict

for dirname, dirnames, filenames in os.walk('./sentenceMatFiles'):
    # # print path to all subdirectories first.
    for filename in filenames:
        fileToLoad = os.path.join('sentenceMatFiles',filename)
        sentences = scipy.io.loadmat(fileToLoad)
        sentences = sentences['sentenceData']
        img = filename[0:(len(filename)-4)] 
        for i in range(len(sentences)):
            current_sentence = sentences[i]
            for j in range(len(current_sentence)):
                chunks = current_sentence[j][1]
                if len(chunks) > 0:
                    # print 'CHUNKS',chunks
                    chunks = chunks[0]
                    mapping = current_sentence[j][3][0]
                    # print 'MAPPING', mapping, 'LEN', len(mapping)
                    # print 'AMOUNT OF SENTENCES', len(chunks)
                    for k in range(len(chunks)):
                        content = chunks[k]
                        content = [item for sublist in content for item in sublist]
                        content = [item for sublist in content for item in sublist]
                        sentence = ' '.join(content).encode('utf-8')
                        annotation_id = mapping[k][0]
                        # print sentence, annotation_id
                        if int(annotation_id) != 0:
                            filename =  img + '_' + annotation_id + '.txt'
                        # print mapping[k][0]
                        # print sentence
                            filetowrite = os.path.join('sentence_snippets', filename)
                            text_file = open(filetowrite, "a")
                            # print img
                            # print sentence
                            text_file.write(sentence+'\n')
                            text_file.close()