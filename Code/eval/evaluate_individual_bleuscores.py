__author__ = 'Wout & thijs'

import argparse
import json
import nltkbleu
import nltk

def main(params):

  # load the result struct
  struct_path = "../Results/"+params['struct']
  ngrams = params['ngrams']
  with open(struct_path) as data_file:
      data = json.load(data_file)
      images = data['imgblobs']
  print_sorted(calculate_single_bleu(images,struct_path,ngrams))
  calculate_total_bleu(images,struct_path,ngrams)

''' Calculates and returns a dictionary containing the bleuscores, candidates and references for each image'''
def calculate_single_bleu(images,struct_path,n):
    results = {}
    for image in images:
        refs = image['references']
        ref_texts = []
        sentences = []
        for ref in refs:
            ref_texts.append(nltk.word_tokenize(ref['text']))
            sentences.append(ref['text'])
        cand = nltk.word_tokenize(image['candidate']['text'])
        bleu = nltkbleu.sentence_bleu(ref_texts,cand,get_weights(n))
        str(sentences)+ image['candidate']['text']+ str(bleu)
        results.update({image['imgid']:{'cand':image['candidate']['text'],'ref':sentences,'bleu':bleu}})
    return results

''' Calculates and prints the total corpus bleu score, note that this isn't just the average'''
def calculate_total_bleu(images,struct_path,ngrams):
    all_candidates = []
    all_references = []
    for image in images:
        refs = image['references']
        ref_texts = []
        for ref in refs:
            ref_texts.append(nltk.word_tokenize(ref['text']))
        cand = nltk.word_tokenize(image['candidate']['text'])
        all_candidates.append(cand)
        all_references.append(ref_texts)
    print 'corpus bleu: ' + str(nltkbleu.corpus_bleu(all_references,all_candidates,get_weights(ngrams)))

''' Saves the list of generated sentences sorted on their bleu score together with their reference sentences.'''
def print_sorted(dict):
    new_list = {}
    for entry in dict.keys():
        new_list.update({entry: dict[entry]['bleu']})
    sorted = sort(new_list)
    f= open('individual_evaluation.txt','w')
    for element in sorted:
        dict_entry = dict.get(element)
        f.write(str(dict_entry['bleu'])+'\n')
        f.write(str(dict_entry['cand'])+'\n')
        f.write('\n')
        f.write(str(dict_entry['ref'])+'\n')
        f.write('---------------------------------\n')
    f.close()

''' Returns the image ids sorted by bleu-score'''
def sort(dict):
    return sorted(dict, key=dict.get)

''' Returns a list of uniform weights, based on the choice of n'''
def get_weights(n):
    value = 1/(n*1.0)
    weights = []
    for i in range(n):
        weights.append(value)
    return weights


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--struct', type=str, default="result_struct.json", help='the result json file')
  parser.add_argument('--ngrams', type=int, default=4, help='the number of ngrams')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
