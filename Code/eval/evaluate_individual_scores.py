__author__ = 'Wout & thijs'

import argparse
import json
from BleuScore import BleuScore
from MeteorScore import MeteorScore


def main(params):

  # load the result struct
  struct_path = "../Results/"+params['struct']
  ngrams = params['ngrams']
  with open(struct_path) as data_file:
      data = json.load(data_file)
      images = data['imgblobs']
  estrategy = getStrategy(params['metric'])
  if(params['individual']==1):
      print_sorted(calculate_individual_score(images,struct_path,ngrams,estrategy))
  calculate_total_score(images,struct_path,ngrams,estrategy)

def getStrategy(metric):
    if(metric == "bleu"):
        return BleuScore("bleu")
    elif(metric == "meteor"):
        return MeteorScore("meteor")
    else:
        return BleuScore("bleu")

def calculate_individual_score(images,struct_path,n,evalStrategy):
    results = {}
    for image in images:
        refs = image['references']
        ref_texts = []
        for ref in refs:
            ref_texts.append(ref['text'])
        cand = image['candidate']['text']
        score = evalStrategy.evaluate_sentence(cand,ref_texts,n)
        str(ref_texts)+ image['candidate']['text']+ str(evalStrategy.get_name())
        results.update({image['imgid']:{'cand':image['candidate']['text'],'ref':ref_texts,'score':score}})
    return results

''' Calculates and prints the total corpus bleu score, note that this isn't just the average'''
def calculate_total_score(images,struct_path,ngrams,evalStrategy):
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


''' Saves the list of generated sentences sorted on their bleu score together with their reference sentences.'''
def print_sorted(dict):
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

''' Returns the image ids sorted by score'''
def sort(dict):
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
