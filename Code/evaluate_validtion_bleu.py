__author__ = 'Wout & thijs'

import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math
import subprocess as sp

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

def main(params):

  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']
  dataset = 'flickr30k'
  # fetch the data provider
  dp = getDataProvider(dataset)
  dp.load_topic_models(dataset, params['lda'])

  best_bleu = 0
  best = ""
  _,_,files = os.walk(checkpoint_path)
  for f in files:

      print 'loading checkpoint %s' % (f, )
      checkpoint = pickle.load(open(checkpoint_path+'/'+f, 'rb'))
      checkpoint_params = checkpoint['params']
      model = checkpoint['model']

      misc = {}
      misc['wordtoix'] = checkpoint['wordtoix']
      ixtoword = checkpoint['ixtoword']


      # iterate over all images in test set and predict sentences
      BatchGenerator = decodeGenerator(checkpoint_params)
      n = 0
      all_references = []
      all_candidates = []

      if params['cca']:
        ccaweights = np.loadtxt('cca/imageprojection_'+str(params['cca'])+'.txt', delimiter = ',')
        misc['ccaweights'] = ccaweights
      else:
        ccaweights = None
      for img in dp.iterImages(split = 'val', max_images = max_images):
        n+=1
        print 'image %d/%d:' % (n, max_images)
        references = [' '.join(x['tokens']) for x in img['sentences']] # as list of lists of tokens
        kwparams = { 'beam_size' : params['beam_size'], 'normalization': params['normalization'], 'ccaweights' : ccaweights }
        if not params['lda'] == 0:
            Ys = BatchGenerator.predict_test([{'image':img}], model, checkpoint_params, **kwparams)
        else:
            Ys = BatchGenerator.predict_test([{'image':img}], model, checkpoint_params, **kwparams)


        for gtsent in references:
          print 'GT: ' + gtsent

        # now evaluate and encode the top prediction
        top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
        top_prediction = top_predictions[0] # these are sorted with highest on top
        candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
        print 'PRED: (%f) %s' % (top_prediction[0], candidate)

        # save for later eval
        all_references.append(references)
        all_candidates.append(candidate)

      # use perl script to eval BLEU score for fair comparison to other research work
      # first write intermediate files
      print 'writing intermediate files into eval/'
      open('eval/output', 'w').write('\n'.join(all_candidates))
      for q in xrange(5):
        open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in all_references]))
      # invoke the perl script to get BLEU scores
      print 'invoking eval/multi-bleu.perl script...'
      owd = os.getcwd()
      os.chdir('eval')
      process = sp.Popen('./multi-bleu.perl reference < output',stdin=sp.PIPE, stdout=sp.PIPE, shell=True)
      lines_iterator = iter(process.stdout.readline, b"")
      for line in lines_iterator:
            fourth = line.split("/")[4]
            bleu_4  = fourth.split(" ")[0]
      if(bleu_4>best_bleu):
          best = f
          best_bleu = bleu_4
      os.chdir(owd)
  print 'Best bleu score for %s : %s' % (best,best_bleu)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('--result_struct_filename', type=str, default='result_struct.json', help='filename of the result struct to save')
  parser.add_argument('-m', '--max_images', type=int, default=-1, help='max images to use')
  parser.add_argument('-d', '--dump_folder', type=str, default="", help='dump the relevant images to a separate folder with this name?')
  parser.add_argument('--lda', type=int, default = 0, help = 'number of topics to be used')
  parser.add_argument('--cca',dest='cca', type=int, default = 0, help = 'number of ccs to be used')
  parser.add_argument('--normalization', type=str, default=None, help='length normalization to be used')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
