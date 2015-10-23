__author__ = 'Thijs'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np
from nltk.stem.porter import *


class TopicExtractor:

    def __init__(self, dataset, nbOfTopics,iterations=1500):
        self.nbOfTopics=nbOfTopics
        self.iterations=iterations
        self.dataprovider = getDataProvider(dataset)
        self.nbOfWordOccurences = 5

    def stopwords(self):
        stopwords = set()
        file=open('lda_images/english')
        for line in file:
            stopwords.update(line)
        return stopwords

    def concatenate_sentences(self):
        current_image=''
        current_sentence=''
        image_sentence_pair_generator = self.dataprovider.iterImageSentencePair(split = 'train')
        output = {}
        for pair in image_sentence_pair_generator:
            sentence = self.remove_common_words(pair['sentence']['tokens'])
            #print sentence
            image = pair['image']['filename']
            #print('image: '+str(pair['image']['filename']))
            if(image is current_image):
                current_sentence=current_sentence+sentence
            else:
                current_image=image
                current_sentence=sentence
            output[image] = current_sentence
        return output

    def remove_common_words(self,sentence):
        s=self.stopwords()
        s.update(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word in s and len(word)>2:
                result.append(word)
        return result

    def model(self, document_term_matrix):
        model = lda.LDA(self.nbOfTopics, n_iter=self.iterations, random_state=1)
        model.fit(document_term_matrix)
        return model

    def stem(self, word):
        stemmer = PorterStemmer()
        return stemmer.stem(word)

    def extract_model(self):
        print('Concatening sentences')
        image_document_pairs = self.concatenate_sentences()
        print('Documents size: '+str(len(image_document_pairs)))
        print('Creating vocabulary')
        vocabulary = self.get_vocabulary(image_document_pairs.values())
        print('Vocabulary size: '+str(len(vocabulary)))
        print('Creating term document matrix')
        matrix = self.create_document_term_matrix(image_document_pairs.values(),vocabulary)
        #self.printMatrix(matrix) #Prints the document-term matrix
        print('Creating model')
        model = self.model(matrix)
        return model,vocabulary,image_document_pairs.keys()

    def printMatrix(self, matrix):
        f = open('matrixFile.txt','w')
        for row in matrix:
            for number in row:
                f.write(str(number)+' ')
            f.write('\n')

    def create_document_term_matrix(self, documents, vocabulary):
        result = np.zeros([len(documents),len(vocabulary)])
        j=0
        for document in documents:
            row = np.zeros(len(vocabulary))
            for word in document:
                if self.stem(word) in vocabulary:
                    i = vocabulary.index(self.stem(word))
                    row[i] += 1
            result[j] = row
            j+=1
        print('type' +str(result.dtype))
        return result.astype(np.int32) #convert to 32bit

    def get_vocabulary(self,documents):
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
        dict = {}
        result = {}
        for document in documents:
            for word in document:
                word = self.stem(word)
                if(not word in dict):
                    dict[word]=1
                else:
                    dict[word]+=1
        for word in dict:
            if(dict[word] >= self.nbOfWordOccurences):
                result[word]=dict[word]
        return result.keys()