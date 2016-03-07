__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import subprocess as sp
import codecs
import nltk

''' METEOR Scores as defined in https://www.cs.cmu.edu/~alavie/METEOR/ '''
class MeteorScore(EvaluationStrategy):

    ''' Evaluates the METEOR score of a single sentence, given its references'''
    def evaluate_sentence(self,sentence,references,n):
        self.write_singlereferences(references)
        self.write_singlesentences(sentence)
        command = "java -Xmx2G -jar meteor/meteor-1.5.jar meteor_sentences.txt meteor_references.txt -l en -norm"
        process = sp.Popen(command,stdin=sp.PIPE, stdout=sp.PIPE, shell=True)
        lines_iterator = iter(process.stdout.readline, b"")
        fline = ""
        for line in lines_iterator:
            fline = line
        result = fline.split("            ")
        return result[1]

    ''' Evaluates the METEOR score of an entire corpus '''
    def evaluate_total(self,sentences,references,n):
         self.write_references(references)
         self.write_sentences(sentences)

         command = "java -Xmx2G -jar meteor/meteor-1.5.jar meteor_sentences.txt meteor_references.txt -l en -norm"
         process = sp.Popen(command,stdin=sp.PIPE, stdout=sp.PIPE, shell=True)
         lines_iterator = iter(process.stdout.readline, b"")
         fline = ""
         for line in lines_iterator:
             print line
             fline = line
         result = fline.split("            ")
         return result[1]

    ''' Writes a list containing lists of 5 references to a txt-file'''
    def write_references(self,references):
         f= open('meteor_references.txt','w')
         for lof_references in references:
             for reference in lof_references:
                 f.write(nltk.word_tokenize(reference)+'\n')
         f.close()

    ''' Writes 5 copies of the sentences in a list to a txt-file '''
    def write_sentences(self,sentences):
        f = open('meteor_sentences.txt','w')
        for sentence in sentences:
            for i in range(5):
                f.write(sentence+'\n')
        f.close()

    ''' Writes one sentence 5 times to a txt-file'''
    def write_singlesentences(self,sentence):
        f = open('meteor_sentences.txt','w')
        for i in range(5):
            f.write(sentence+'\n')
        f.close()

    ''' Writes a list of references to a txt-file '''
    def write_singlereferences(self,sentences):
        f = open('meteor_references.txt','w')
        for sentence in sentences:
            f.write(sentence+'\n')
        f.close()