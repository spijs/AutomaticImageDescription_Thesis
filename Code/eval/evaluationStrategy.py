__author__ = 'spijs'

class EvaluationStrategy(object):
    def __init__(self,name):
        self.name = name

    def evaluate_sentence(self,sentence,references,n):
        pass

    def evaluate_total(self,sentences,references,n):
        pass

    def get_name(self):
        return self.name