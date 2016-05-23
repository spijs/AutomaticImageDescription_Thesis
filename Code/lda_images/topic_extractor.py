__author__ = 'Wout & Thijs'

import lda
from imagernn.data_provider import getDataProvider
import numpy as np
from nltk.stem.porter import *

'''This class uses lda to extract topics from images based on their description'''
class TopicExtractor:

    def __init__(self, dataset, nbOfTopics,iterations=1500, pert = None):
        self.nbOfTopics=nbOfTopics
        self.iterations=iterations
        self.dataprovider = getDataProvider(dataset, pert)
        self.nbOfWordOccurences = 5


    def stopwords(self):
        '''
        Returns a list containing all the considered english stopwords
        '''
        stopwords = set()
        file=open('lda_images/english')
        for line in file.readlines():
            stopwords.add(line[:-1])
        return stopwords

    def concatenate_sentences(self):
        '''
        Concatenates the sentences for each image in each split to one string. Returns the resulting image string pairs for each
        split
        :return: dictionary mapping split name to another dictionary. These three dictionaries map image names to
        the concatenation of the corresponding sentences
        '''
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


    def remove_common_words(self,sentence,stopwords):
        '''
        Removes the stopwords and words with length <3 from a given sentence
        :param sentence: sentence to process
        :param stopwords: list of stopwords to remove
        :return: given sentence, without stopwords and words shorter than 3 characters
        '''
        stopwords.add(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word.lower() in stopwords and len(word)>2:
                result.append(word.lower())
        return result

    def model(self, document_term_matrix):
        '''
        :param document_term_matrix: the document term matrix to use
        :return: LDA model based on the given document term matrix
        '''
        model = lda.LDA(self.nbOfTopics, n_iter=self.iterations, random_state=1)
        model.fit(document_term_matrix)
        return model

    def stem(self, word):
        '''
        :return: given word, stemmed using Porter stemming algorithm
        '''
        stemmer = PorterStemmer()
        return stemmer.stem(word)

    def extract_model(self):
        '''
        Creates splits, creates vocabulary for the training set, creates the document term matrix for training set
        and finally creates a model with this information
        :return: the LDA model, the training vocabulary and dictionaries mapping image names to the concatenation
        of the corresponding sentences
        '''
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
        print('Creating model')
        model = self.model(matrix)
        return model,vocabulary,splitPairs

    def create_document_term_matrix(self, documents, vocabulary):
        '''
        :param documents
        :param vocabulary
        :return: document term matrix based on the given documents and vocabulary
        '''
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
        '''
        stem all the words in the given documents and return the used vocabulary
        :param documents: list of documents
        :return: vocabulary based on the stemmed version of the given documents
        '''
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
