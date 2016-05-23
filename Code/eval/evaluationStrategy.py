__author__ = 'Thijs&Wout'

class EvaluationStrategy(object):
    '''
    Strategy for evaluation a set of generated sentences.
    '''
    def __init__(self,name):
        #TODO should be abstract
        self.name = name

    def evaluate_sentence(self,sentence,references,n):
        pass

    def evaluate_total(self,sentences,references,n):
        pass

    def get_name(self):
        return self.name