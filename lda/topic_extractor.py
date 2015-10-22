__author__ = 'Thijs'

import lda
from nltk.corpus import stopwords
from imagernn.data_provider import getDataProvider
from numpy import empty, zeros

class TopicExtractor:

    def __init__(self, dataset, nbOfTopics,iterations=1500):
        self.nbOfTopics=nbOfTopics
        self.iterations=iterations
        self.dataprovider = getDataProvider(dataset)

    def concatenate_sentences(self):
        current_image=''
        current_sentence=''
        image_sentence_pair_generator = self.dataprovider.iterImageSentencePair()
        output = {}
        for pair in image_sentence_pair_generator:
            sentence = self.remove_common_words(pair['sentence'])
            if(pair['image'] is current_image):
                current_sentence=current_sentence+sentence
            else:
                current_image=pair['image']
                current_sentence=sentence
            output['image'] = current_sentence
        return output

    #TODO hangt af van wat ik binnenkrijg als sentence :D, ik veronderstel lijst denk ik
    def remove_common_words(self,sentence):
        s=set(stopwords.words('english'))
        output_sentence = [" ".join([w for w in t.split() if not w in s]) for t in sentence]
        return output_sentence

    def model(self, document_term_matrix):
        model = lda.LDA(self.nbOfTopics, n_iter=self.iterations, random_state=1)
        model.fit(document_term_matrix)
        return model

    def extract_model(self):
        image_document_pairs = self.concatenate_sentences()
        vocabulary = self.get_vocabulary(image_document_pairs.values())
        matrix = self.create_document_term_matrix(image_document_pairs.values(),vocabulary)
        model = self.model(matrix)
        return model

    def create_document_term_matrix(self, documents, vocabulary):
        result = empty([len(documents),len(vocabulary)])
        j=0
        for document in documents:
            row = zeros(len(vocabulary))
            for word in document:
                i = vocabulary.index(str(word))
                row[i] += 1
            result[j] = row
        return result

    def get_vocabulary(self,documents):
    # count up all word counts so that we can threshold
    # this shouldnt be too expensive of an operation
        dict = {}
        for document in documents:
            for word in document:
                dict[word]=1
        return dict.keys()