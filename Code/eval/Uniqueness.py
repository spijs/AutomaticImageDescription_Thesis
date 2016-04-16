__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import subprocess as sp
import codecs
import nltk
import json


class Uniqueness(EvaluationStrategy):

    def __init__(self,name):
        self.train_sentences  = self.load_sentences()
        print self.train_sentences
        super(Uniqueness,self).__init__(name)

    def load_sentences(self):
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

    ''' Evaluates the METEOR score of a single sentence, given its references'''
    def evaluate_sentence(self,sentence,references,n):
        pass

    ''' Evaluates the METEOR score of an entire corpus '''
    def evaluate_total(self,sentences,references,n):
         nb_of_unique=0
         for s in sentences:
            if self.is_unique(s):
                nb_of_unique += 1
         return nb_of_unique

    def is_unique(self,sentence):
        new_words = sentence.split(" ")
        for train in self.train_sentences:
            if self.is_same_sentence(new_words,train):
                return False
        return True

    def is_same_sentence(self,s1,s2):
        if len(s1) != len(s2):
                return False
        for i in range(len(s2)):
            if s1[i] != s2[i]:
                return False
        return True