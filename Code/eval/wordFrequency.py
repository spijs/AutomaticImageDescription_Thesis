__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import nltk

class WordFrequency(EvaluationStrategy):

    ''' Evaluates and returns the sorted word frequencies of generated sentences    '''
    def evaluate_total(self,sentences,references,n):
        words_in_sentences = {}
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for word in words:
                if words_in_sentences.get(word):
                    words_in_sentences[word] = words_in_sentences[word]+1
                else:
                    words_in_sentences[word] = 1
        sorted_words = sorted(words_in_sentences, key=words_in_sentences.get)
        for word in reversed(sorted_words):
            print word + " " + str(words_in_sentences[word])
        nb = self.count_unique_words(references)
        print("Number of sentences: "+str(len(sentences)))
        print "Average sentence length: " + str(self.get_average_sentence_length(sentences))
        print "Number of unique words in references: " + str(nb)
        return len(words_in_sentences)

    def get_average_sentence_length(self,sentences):
        sum = 0.0
        distribution = {}
        for sentence in sentences:
            nb = len(nltk.word_tokenize(sentence))
            sum += nb
            if distribution.get(nb):
                distribution[nb] = distribution[nb]+1
            else:
                distribution[nb] = 1
        for el in distribution.keys():
            print "Number of length "+str(el)+ "sentences :"+str(distribution[el])
        return sum/len(sentences)

    ''' count the number of unique words in a list of lists of sentences '''
    def count_unique_words(self,references):
         dict = []
         for reflist in references:
            for ref in reflist:
                words = nltk.word_tokenize(ref)
                for word in words:
                    if not word in dict:
                        dict.append(word)
         return len(dict)

    ''' Returns a list of uniform weights, based on the choice of n'''
    def get_weights(self,n):
        value = 1/(n*1.0)
        weights = []
        for i in range(n):
            weights.append(value)
        return weights