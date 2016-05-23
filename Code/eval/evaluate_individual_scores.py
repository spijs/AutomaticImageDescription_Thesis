__author__ = 'Wout & Thijs'

import argparse
import json
from BleuScore import BleuScore
from MeteorScore import MeteorScore
from wordFrequency import WordFrequency
from Uniqueness import Uniqueness


def main(params):
    '''
    Evaluates the test set with the given parameters.
    :param params: parameters for evaluating
    :return: None (prints the results)
    '''
    # load the result struct
    struct_path = "../Results/"+params['struct']
    ngrams = params['ngrams']
    with open(struct_path) as data_file:
        data = json.load(data_file)
        images = data['imgblobs']
    estrategy = getStrategy(params['metric'])
    if(params['individual']==1):
        print_sorted(calculate_individual_score(images,struct_path,ngrams,estrategy))
    else:
        calculate_total_score(images,struct_path,ngrams,estrategy)

def getStrategy(metric):
    '''
    Returns a strategy for evaluating based on the given parameter
    :param metric: metric to be used as a string
    :return: EvaluationStrategy
    '''
    if(metric == "bleu"):
        return BleuScore("bleu")
    elif(metric == "meteor"):
        return MeteorScore("meteor")
    elif(metric == "freq"):
        return WordFrequency("freq")
    elif(metric == "unique"):
        return Uniqueness("unique")
    else:
        return BleuScore("bleu")


def calculate_individual_score(images,struct_path,n,evalStrategy):
    '''
    Calculates the score of individual sentences
    :param images: dictionary of images that need to be evaluated
    :param struct_path: path containing the json-file that needs to be evaluated
    :param n: number of n-grams
    :param evalStrategy: Evaluation strategy to be used.
    :return: results on each individual sentence
    '''
    results = {}
    bleu = 0
    for image in images:
        refs = image['references']
        ref_texts = []
        for ref in refs:
            ref_texts.append(ref['text'])
        cand = image['candidate']['text']
        score = evalStrategy.evaluate_sentence(cand,ref_texts,n)
        str(ref_texts)+ image['candidate']['text']+ str(evalStrategy.get_name())
        results.update({image['imgid']:{'cand':image['candidate']['text'],'ref':ref_texts,'score':score}})
        bleu+=score
    print "Total average bleu : " + str(1.0*bleu/(len(images)))
    return results

def calculate_total_score(images,struct_path,ngrams,evalStrategy):
    '''
    Calculates the score for an entire corpus.
    (Note that this is not necessarily equal to the average of individual scores)
    :param images: dictionary of images that need to be evaluated
    :param struct_path: path containing the json-file that needs to be evaluated
    :param n: number of n-grams
    :param evalStrategy: Evaluation strategy to be used.
    :return: None (prints the results of entire corpus.)
    '''
    all_candidates = []
    all_references = []
    for image in images:
        refs = image['references']
        ref_texts = []
        for ref in refs:
            ref_texts.append(ref['text'])
        cand = image['candidate']['text']
        all_candidates.append(cand)
        all_references.append(ref_texts)
    score = evalStrategy.evaluate_total(all_candidates,all_references,ngrams)
    print 'corpus score: ' + str(score)


def print_sorted(dict):
    '''
    Saves the list of generated sentences sorted on their individual score together with their reference sentences.
    :param dict: dictionary of sentences and scores
    :return:None (writes sorted generated sentences to individual_evaluation.txt)
    '''
    new_list = {}
    for entry in dict.keys():
        new_list.update({entry: dict[entry]['score']})
    sorted = sort(new_list)
    f= open('individual_evaluation.txt','w')
    for element in sorted:
        dict_entry = dict.get(element)
        f.write(str(dict_entry['score'])+'\n')
        f.write(str(dict_entry['cand'])+'\n')
        f.write('\n')
        f.write(str(dict_entry['ref'])+'\n')
        f.write('---------------------------------\n')
    f.close()

def sort(dict):
    ''' Returns the image ids sorted by score'''
    return sorted(dict, key=dict.get)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--struct', type=str, default="result_struct.json", help='the result json file')
  parser.add_argument('--ngrams', type=int, default=4, help='the number of ngrams')
  parser.add_argument('--metric', type=str, default="bleu", help='the metric to evaluate the generated sentences')
  parser.add_argument('--individual', type=int, default=0, help='do you want individually ordered results? Yes=1,No=0=default')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
