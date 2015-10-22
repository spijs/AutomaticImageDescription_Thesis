__author__ = 'Thijs'

import lda
from nltk.corpus import stopwords
from imagernn.data_provider import getDataProvider
from numpy import zeros,sum

class TopicExtractor:

    def __init__(self, dataset, nbOfTopics,iterations=1500):
        self.nbOfTopics=nbOfTopics
        self.iterations=iterations
        self.dataprovider = getDataProvider(dataset)
        self.nbOfWordOccurences = 5

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
    
    #TODO misschien stemming?
    def remove_common_words(self,sentence):
        s=set(stopwords.words('english'))
        s.update(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word in s:
                result.append(word)
        return result

    def model(self, document_term_matrix):
        model = lda.LDA(self.nbOfTopics, n_iter=self.iterations, random_state=1)
        model.fit(document_term_matrix)
        return model

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
        return model

    def printMatrix(self, matrix):
        f = open('matrixFile.txt','w')
        for row in matrix:
            for number in row:
                f.write(str(number)+' ')
            f.write('\n')

    def create_document_term_matrix(self, documents, vocabulary):
        result = zeros([len(documents),len(vocabulary)])
        j=0
        for document in documents:
            row = zeros(len(vocabulary))
            for word in document:
                if word in vocabulary:
                    i = vocabulary.index(word)
                    row[i] += 1.0
            result[j] = row
            j+=1
        return result

    def get_vocabulary(self,documents):
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
        dict = {}
        result = {}
        for document in documents:
            for word in document:
                if(not word in dict):
                    dict[word]=1
                else:
                    dict[word]+=1
        for word in dict:
            if(dict[word] >= self.nbOfWordOccurences):
                result[word]=dict[word]
        return result.keys()