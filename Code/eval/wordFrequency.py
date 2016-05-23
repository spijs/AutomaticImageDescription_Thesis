__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import nltk

class WordFrequency(EvaluationStrategy):
    '''
    Evaluates different statistics of the generated sentence.
    '''

    def evaluate_total(self,sentences,references,n):
        ''' Evaluates and prints the sorted word frequencies of generated sentences
            prints each word in the generated sentences together with their frequency.
            prints the average sentence length
            prints the number of unique words in the references
            returns the number of unique words in the sentences.
         '''
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
        print "Average sentence length: " + str(self.get_average_sentence_length(sentences))
        print "Number of unique words in references: " + str(nb)
        return len(words_in_sentences)

    def get_average_sentence_length(self,sentences):
        '''
        Calculates the average sentence length in the provided sentences.
        '''
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

    def count_unique_words(self,references):
        ''' count the number of unique words in a list of lists of sentences '''
        dict = []
        for reflist in references:
            for ref in reflist:
                words = nltk.word_tokenize(ref)
                for word in words:
                    if not word in dict:
                        dict.append(word)
        return len(dict)