__author__ = 'spijs'

from evaluationStrategy import EvaluationStrategy
import nltk
import nltkbleu

class BleuScore(EvaluationStrategy):
    '''BLEU evaluation of the generated sentences
    -> should ONLY be used for evaluating and sorting individual sentences.
    multi-bleu.perl should be used for evluating full corpus'''

    def evaluate_sentence(self,sentence,references,n):
        ''' Evaluates and returns the bleu score of a single sentence given its references and n of ngrams    '''
        ref_texts = []
        weights = self.get_weights(n)
        for ref in references:
            ref_texts.append(nltk.word_tokenize(ref))
        cand = nltk.word_tokenize(sentence)
        bleu = 0
        #sf = nltkbleu.SmoothingFunction()
        if cand and len(cand)>0:
            bleu = nltkbleu.sentence_bleu(ref_texts,cand,weights)
            if bleu == 0:
                print "candidate :" + str(cand)
                print "references : "+ str(ref_texts)

        return bleu


    def evaluate_total(self,sentences,references,n):
        ''' Evaluates and returns the bleu score of a corpus of sentences given their references and n of ngrams
            SHOULD NOT BE USED'''
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


    def get_weights(self,n):
        ''' Returns a list of uniform weights, based on the choice of n'''
        value = 1/(n*1.0)
        weights = []
        for i in range(n):
            weights.append(value)
        return weights