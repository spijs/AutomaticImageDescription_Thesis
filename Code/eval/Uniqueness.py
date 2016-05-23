__author__ = 'Thijs & Wout'

from evaluationStrategy import EvaluationStrategy
import subprocess as sp
import codecs
import nltk
import json


class Uniqueness(EvaluationStrategy):

    def __init__(self,name):
        self.train_sentences  = self.load_sentences()
        super(Uniqueness,self).__init__(name)

    def load_sentences(self):
        '''
        Loads and returns the sentences from the training set
        '''
        dataset = json.load(open('../data/flickr30k/dataset.json', 'r'))
        sentence_tokens =[]
        # group images by their train/val/test split into a dictionary -> list structure
        for img in dataset['images']:
            split = img['split']
            sentences = img['sentences']
            for sentence in sentences:
                tokens = sentence['tokens']
                if split=='train':
                    sentence_tokens.append(tokens)
        return sentence_tokens

    def evaluate_sentence(self,sentence,references,n):
        ''' Not implemented'''
        pass

    def evaluate_total(self,sentences,references=None,n=None):
        ''' Evaluates the uniqueness score of an entire corpus '''
        nb_of_unique=0
        i = 0
        for s in sentences:
            i += 1
            print 'Currently at sentence: %i' % i
            if self.is_unique(s):
                nb_of_unique += 1
        return nb_of_unique

    def is_unique(self,sentence):
        '''
        :param sentence: Sentence to check
        :return: True if this sentence does not occur in the training set.
        '''
        new_words = sentence.split(" ")
        for train in self.train_sentences:
            if self.is_same_sentence(new_words,train):
                return False
        #self.train_sentences.append(new_words) # Append this sentence to all the generated sentences (Uncomment this for Unique2)
        return True

    def is_same_sentence(self,s1,s2):
        '''
        :return: True if s1 and s2 are the same sentence
        '''
        if len(s1) != len(s2):
                return False
        for i in range(len(s2)):
            if s1[i] != s2[i]:
                return False
        return True