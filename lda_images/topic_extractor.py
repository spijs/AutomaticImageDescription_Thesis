__author__ = 'Thijs'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np
from nltk.stem.porter import *
from nltk.corpus import stopwords

'''This class uses lda to extract topics from images based on their description'''
class TopicExtractor:

    def __init__(self, dataset, nbOfTopics,iterations=1500):
        self.nbOfTopics=nbOfTopics
        self.iterations=iterations
        self.dataprovider = getDataProvider(dataset, nbOfTopics)
        self.nbOfWordOccurences = 5 #TODO niet langer hardcoden?

    ''' Returns a list containing all the considered english stopwords'''
    def stopwords(self):
        stopwords = set()
        file=open('lda_images/english')
        for line in file.readlines():
            stopwords.add(line[:-1])
        return stopwords

    ''' Concatenates the sentences for each image in each split to one string. Returns the resulting image string pairs for each
        split'''
    def concatenate_sentences(self):
        current_image=''
        current_sentence=''
        s=self.stopwords()
        outputsplit = {}
        splits = ['train','test','val']
        #Iterate over each split
        for split in splits:
            output = {}
            #get all the image sentence pairs in the split
            image_sentence_pair_generator = self.dataprovider.iterImageSentencePair(split = split)
            for pair in image_sentence_pair_generator:
                sentence = self.remove_common_words(pair['sentence']['tokens'],s)
                image = pair['image']['filename']
                #Concatenate the sentences
                if(image is current_image):
                    current_sentence=current_sentence+sentence
                else:
                    current_image=image
                    current_sentence=sentence
                output[image] = current_sentence
            outputsplit[split] = output
        return outputsplit


    ''' Removes the stopwords and words with length =< 2 from a given sentence'''
    def remove_common_words(self,sentence,stopwords):
        #s = set(stopwords.words('english'))
        stopwords.add(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word.lower() in stopwords and len(word)>2:
                result.append(word.lower())
        return result

    ''' Creates and returns an lda model for the given document term matrix'''
    def model(self, document_term_matrix):
        model = lda.LDA(self.nbOfTopics, n_iter=self.iterations, random_state=1)
        model.fit(document_term_matrix)
        return model

    ''' stems a word by using the porter algorithm'''
    def stem(self, word):
        stemmer = PorterStemmer()
        return stemmer.stem(word)

    ''' Creates splits, creates vocabulary for the training set, creates the document term matrix for training set
        and finally creates a model with this information.'''
    def extract_model(self):
        print('Concatening sentences')
        splitPairs = self.concatenate_sentences()
        train_pairs = splitPairs['train']
        print "length train", len(train_pairs)
        test_pairs = splitPairs['test']
        print "length test", len(test_pairs)
        val_pairs = splitPairs['val']
        print "length val", len(val_pairs)
        print('Documents size: '+str(len(train_pairs)))
        print('Creating vocabulary')
        #Create vocabulary using only the train data
        vocabulary = self.get_vocabulary(train_pairs.values())
        print('Vocabulary size: '+str(len(vocabulary)))
        print('Creating term document matrix')
        #Create term document matrix using only the train data
        matrix = self.create_document_term_matrix(train_pairs.values(),vocabulary)
        #self.printMatrix(matrix) #Prints the document-term matrix
        print('Creating model')
        model = self.model(matrix)
        return model,vocabulary,splitPairs

    ''' Pretty print for a matrix into a new file'''
    def printMatrix(self, matrix):
        f = open('matrixFile.txt','w') # Output file
        for row in matrix:
            for number in row:
                f.write(str(number)+' ')
            f.write('\n')

    ''' Creates a term document matrix using a given vocabulary and provided documents'''
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

    ''' Creates a vocabulary given a list of documents'''
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
            if(dict[word] >= self.nbOfWordOccurences): # Only words occurring more than nbOfWordOccurences are entered in vocabulary
                result[word]=dict[word]
        return result.keys()
