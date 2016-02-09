__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import subprocess as sp

class MeteorScore(EvaluationStrategy):

    def evaluate_sentence(self,sentence,references,n):
        pass

    def evaluate_total(self,sentences,references,n):
         self.write_references(references)
         self.write_sentences(sentences)

         command = "java -Xmx2G -jar meteor-1.5.jar meteor_sentences.txt meteor_references.txt -l en -norm"
         process = sp.Popen(command,stdin=sp.PIPE, stdout=sp.PIPE, shell=True)
         lines_iterator = iter(process.stdout.readline, b"")
         for line in lines_iterator:
             print line

    def write_references(self,references):
         f= open('meteor_references.txt','w')
         for lof_references in references:
             for reference in lof_references:
                 f.write(reference+'\n')
         f.close()

    def write_sentences(self,sentences):
        f = open('meteor_sentences.txt','w')
        for sentence in sentences:
            for i in range(5):
                f.write(sentence+'\n')

    ''' Returns a list of uniform weights, based on the choice of n'''
    def get_weights(self,n):
        value = 1/(n*1.0)
        weights = []
        for i in range(n):
            weights.append(value)
        return weights