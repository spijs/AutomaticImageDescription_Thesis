__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import nltk
import nltkbleu

class BlueScore(EvaluationStrategy):
    def evaluate_sentence(self,sentence,references,n):
         ref_texts = []
         for ref in references:
            ref_texts.append(nltk.word_tokenize(ref))
         cand = nltk.word_tokenize(sentence)
         bleu = nltkbleu.sentence_bleu(ref_texts,cand,self.get_weights(n))
         return bleu

    def evaluate_total(self,sentences,references,n):
        candidates = []
        final_references = []
        for sentence in sentences:
            candidates.append(nltk.word_tokenize(sentence))
        for imagerefs in references:
            tokenizedrefs = []
            for imageref in imagerefs:
                tokenizedrefs.append(nltk.word_tokenize(imageref))
            final_references.append(tokenizedrefs)

        return nltkbleu.corpus_bleu(final_references,candidates,self.get_weights(n))

    ''' Returns a list of uniform weights, based on the choice of n'''
    def get_weights(self,n):
        value = 1/(n*1.0)
        weights = []
        for i in range(n):
            weights.append(value)
        return weights