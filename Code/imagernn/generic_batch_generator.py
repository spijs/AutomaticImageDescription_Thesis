__author__ = 'Wout & thijs'

from argparse import _ActionsContainer
import numpy as np
import code
from imagernn.utils import merge_init_structs, initw, accumNpDicts
from imagernn.lstm_generator import LSTMGenerator
from imagernn.rnn_generator import RNNGenerator
from imagernn.fsmn_generator import FSMNGenerator
from imagernn.glstm_generator import gLSTMGenerator
from imagernn.guide import get_guide_size,get_guide

def decodeGenerator(generator):
  if generator == 'lstm':
    return LSTMGenerator
  if generator == 'rnn':
    return RNNGenerator
  if generator == 'fsmn':
    return FSMNGenerator
  if generator == 'glstm':
    return gLSTMGenerator
  else:
    raise Exception('generator %s is not yet supported' % (base_generator_str,))

class GenericBatchGenerator:
  """ 
  Base batch generator class. 
  This class is aware of the fact that we are generating
  sentences from images.
  """

  @staticmethod
  def init(params, misc):

    # inputs
    image_encoding_size = params.get('image_encoding_size', 128)
    word_encoding_size = params.get('word_encoding_size', 128)
    hidden_size = params.get('hidden_size', 128)
    generator = params.get('generator', 'lstm')
    vocabulary_size = len(misc['wordtoix'])
    lda = params.get('lda',0)
    output_size = len(misc['ixtoword']) # these should match though
    image_size = 4096 # size of CNN vectors hardcoded here
    guide_input = params.get('guide',"image")
    if generator == 'lstm':
      assert image_encoding_size == word_encoding_size, 'this implementation does not support different sizes for these parameters'

    # initialize the encoder models
    model = {}
    model['We'] = initw(image_size, image_encoding_size) # image encoder
    model['be'] = np.zeros((1,image_encoding_size))
    model['Ws'] = initw(vocabulary_size, word_encoding_size) # word encoder

    update = ['We', 'be', 'Ws']
    regularize = ['We', 'Ws']

    # Added for LDA
    if lda:
      model['Wlda'] = initw(lda,image_encoding_size)
      update.append('Wlda')
      regularize.append('Wlda')

    init_struct = { 'model' : model, 'update' : update, 'regularize' : regularize}

    # descend into the specific Generator and initialize it
    Generator = decodeGenerator(generator)
    #ADDED
    if(generator != 'glstm'):
        generator_init_struct = Generator.init(word_encoding_size, hidden_size, output_size)
    else:
        guide_size = get_guide_size(guide_input,params['lda'])
        generator_init_struct = Generator.init(word_encoding_size, hidden_size, guide_size, output_size)
    merge_init_structs(init_struct, generator_init_struct)
    return init_struct

  @staticmethod
  def forward(batch, model, params, misc, predict_mode = False):
    """ iterates over items in the batch and calls generators on them """
    # we do the encoding here across all images/words in batch in single matrix
    # multiplies to gain efficiency. The RNNs are then called individually
    # in for loop on per-image-sentence pair and all they are concerned about is
    # taking single matrix of vectors and doing the forward/backward pass without
    # knowing anything about images, sentences or anything of that sort.

    # encode all images
    # concatenate as rows. If N is number of image-sentence pairs,
    # F will be N x image_size
    F = np.row_stack(x['image']['feat'] for x in batch)
    We = model['We']
    be = model['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size
    lda_enabled = params.get('lda',0)
    L = np.zeros((len(batch),lda_enabled))
    if lda_enabled!=0:
      Wlda = model['Wlda']
      L = np.row_stack(x['topics'] for x in batch)
      lda = L.dot(Wlda)

    # decode the generator we wish to use
    generator_str = params.get('generator', 'lstm')
    Generator = decodeGenerator(generator_str)

    guide_input = params.get('guide',None)
    # encode all words in all sentences (which exist in our vocab)
    wordtoix = misc['wordtoix']
    Ws = model['Ws']
    gen_caches = []
    Ys = [] # outputs
    for i,x in enumerate(batch):
      # take all words in this sentence and pluck out their word vectors
      # from Ws. Then arrange them in a single matrix Xs
      # Note that we are setting the start token as first vector
      # and then all the words afterwards. And start token is the first row of Ws
      ix = [0] + [ wordtoix[w] for w in x['sentence']['tokens'] if w in wordtoix ]
      Xs = np.row_stack( [Ws[j, :] for j in ix] )
      Xi = Xe[i,:]
      guide = get_guide(guide_input,F[i,:],L=L[i,:])
      if lda_enabled!=0 and not guide_input:
        guide = lda[i,:]
      # forward prop through the RNN
      gen_Y, gen_cache = Generator.forward(Xi, Xs,guide, model, params, predict_mode = predict_mode)
      gen_caches.append((ix, gen_cache))
      Ys.append(gen_Y)

    # back up information we need for efficient backprop
    cache = {}
    if not predict_mode:
      # ok we need cache as well because we'll do backward pass
      cache['gen_caches'] = gen_caches
      cache['Xe'] = Xe
      if lda_enabled:
        cache['lda'] = lda
      cache['Ws_shape'] = Ws.shape
      cache['F'] = F
      cache['L'] = L
      cache['generator_str'] = generator_str
      cache['lda_enabled'] = lda_enabled
      cache['guide'] = guide_input

    return Ys, cache


  @staticmethod
  def backward(dY, cache):
    Xe = cache['Xe']
    guide = cache['guide']
    generator_str = cache['generator_str']
    dWs = np.zeros(cache['Ws_shape'])
    gen_caches = cache['gen_caches']
    F = cache['F']
    L = cache ['L']
    dXe = np.zeros(Xe.shape)
    lda_enabled = cache['lda_enabled']
    if lda_enabled:
      lda = cache['lda']
      dlda = np.zeros(lda.shape)

    Generator = decodeGenerator(generator_str)

    # backprop each item in the batch
    grads = {}
    for i in xrange(len(gen_caches)):
      ix, gen_cache = gen_caches[i] # unpack
      local_grads = Generator.backward(dY[i], gen_cache)
      dXs = local_grads['dXs'] # intercept the gradients wrt Xi and Xs
      del local_grads['dXs']
      dXi = local_grads['dXi']
      del local_grads['dXi']

      if(lda_enabled and not guide):
        dLi  = local_grads['dLi']
        del local_grads['dLi']
        dlda[i,:] += dLi

      accumNpDicts(grads, local_grads) # add up the gradients wrt model parameters

      # now backprop from dXs to the image vector and word vectors
      dXe[i,:] += dXi # image vector
      for n,j in enumerate(ix): # and now all the other words
        dWs[j,:] += dXs[n,:]

    # finally backprop into the image encoder
    dWe = F.transpose().dot(dXe)
    dbe = np.sum(dXe, axis=0, keepdims = True)

    if lda_enabled:
      dWlda = L.transpose().dot(dlda)
      accumNpDicts(grads, { 'We':dWe, 'be':dbe, 'Ws':dWs, 'Wlda':dWlda})#'Wlda':dWlda
    else:
      accumNpDicts(grads, { 'We':dWe, 'be':dbe, 'Ws':dWs})#'Wlda':dWlda
    return grads

  @staticmethod
  def predict(batch, model, params, **kwparams):
    """ some code duplication here with forward pass, but I think we want the freedom in future """
    F = np.row_stack(x['image']['feat'] for x in batch)
    lda_enabled = params.get('lda',0)
    L = np.zeros((params.get('image_encoding_size',128),lda_enabled))
    if lda_enabled:
       L = np.row_stack(x['topics'] for x in batch)
    We = model['We']
    try:
        Wlda = model['Wlda']
        lda = L.dot(Wlda)
    except(KeyError):
        print 'no wlda'

    be = model['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size
    generator_str = params['generator']
    Generator = decodeGenerator(generator_str)
    Ys = []
    guide_input = params.get('guide','image')
    for i,x in enumerate(batch):
      Xi = Xe[i,:]
      guide = get_guide(guide_input,F[i,:],L=L[i,:])
      if lda_enabled and not guide_input:
        guide = lda[i,:]
      gen_Y = Generator.predict(Xi, guide, model, model['Ws'], params, **kwparams)
      Ys.append(gen_Y)
    return Ys

  @staticmethod
  def predict_test(batch, model, params,  **kwparams):
    """ some code duplication here with forward pass, but I think we want the freedom in future """
    F = np.row_stack(x['image']['feat'] for x in batch)
    lda_enabled = params.get('lda',0)
    L = np.zeros((params.get('image_encoding_size',128),lda_enabled))
    if lda_enabled:
       L = np.row_stack(x['image']['topics'] for x in batch)
    We = model['We']
    Wlda = model['Wlda']
    be = model['be']
    Xe = F.dot(We) + be # Xe becomes N x image_encoding_size
    #print('L shape', L.shape)
    #print('Wlda shape', Wlda.shape)
    lda = L.dot(Wlda)
    generator_str = params['generator']
    Generator = decodeGenerator(generator_str)
    Ys = []
    guide_input = params.get('guide','image')
    for i,x in enumerate(batch):
      Xi = Xe[i,:]
      guide = get_guide(guide_input,F[i,:],L=L[i,:])
      if lda_enabled and not guide_input:
        guide = lda[i,:]
      gen_Y = Generator.predict(Xi, guide, model, model['Ws'], params, **kwparams)
      Ys.append(gen_Y)
    return Ys

