__author__ = 'Karpathy - Modifications by Wout & Thijs'

import numpy as np

from imagernn.utils import initw
from imagernn.Hlayer import HLayer


class FSMNGenerator:
    """
  An FSMN generator.
  This class is as stupid as possible. Basic implementation.
  Implemented backpropagation M arbitrarily to check for speed-up.
  """

    @staticmethod
    def init(input_size, hidden_size, output_size, layers):
        model = {}
        # connections to x_t
        model['Wxh'] = initw(input_size, hidden_size)
        model['bxh'] = np.zeros((1, hidden_size))
        # connections to hmem_{t-1}

        '''model['Whh'] = initw(hidden_size, hidden_size)
        '''
        model['bhh'] = np.zeros((1, hidden_size))
        # Decoder weights (e.g. mapping to vocabulary)
        model['Wd'] = initw(hidden_size, output_size) * 0.1  # decoder
        model['bd'] = np.zeros((1, output_size))


        update = ['Wxh', 'bxh', 'Wd', 'bd','bhh'] #,'bhh','Whh
        regularize = ['Wxh', 'Wd']

        N = 8  # TODO change so it is no longer hardcoded
        for l in range(layers):
            model['Whh'+str(l)] = initw(hidden_size, hidden_size)
            model['bhh'+str(l)] = np.zeros((1, hidden_size))
            model['A'+str(l)] = initw(1,N)
            update.append('Whh'+str(l))
            update.append('bhh'+str(l))
            update.append('A'+str(l))
            regularize.append('Whh'+str(l))
        return {'model': model, 'update': update, 'regularize': regularize}

    @staticmethod
    def forward(Xi, Xs, model, params, **kwargs):
        """
    Xi is 1-d array of size D1 (containing the image representation)
    Xs is N x D2 (N time steps, rows are data containng word representations), and
    it is assumed that the first row is already filled in as the start token. So a
    sentence with 10 words will be of size 11xD2 in Xs.
    """
        predict_mode = kwargs.get('predict_mode', False)

        # options
        drop_prob_encoder = params.get('drop_prob_encoder', 0.0)
        drop_prob_decoder = params.get('drop_prob_decoder', 0.0)

        relu_encoders = params.get('rnn_relu_encoders', 0)
        '''
        rnn_feed_once = params.get('rnn_feed_once', 0)
        '''

        if drop_prob_encoder > 0: # if we want dropout on the encoder
      # inverted version of dropout here. Suppose the drop_prob is 0.5, then during training
      # we are going to drop half of the units. In this inverted version we also boost the activations
      # of the remaining 50% by 2.0 (scale). The nice property of this is that during prediction time
      # we don't have to do any scaling, since all 100% of units will be active, but at their base
      # firing rate, giving 100% of the "energy". So the neurons later in the pipeline dont't change
      # their expected firing rate magnitudes
            if not predict_mode: # and we are in training mode
                scale = 1.0 / (1.0 - drop_prob_encoder)
                Us = (np.random.rand(*(Xs.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
                Xs *= Us # drop!
                Ui = (np.random.rand(*(Xi.shape)) < (1 - drop_prob_encoder)) * scale
                Xi *= Ui # drop!

        # encode input vectors
        Wxh = model['Wxh']
        bxh = model['bxh']
        bhh = model ['bhh']
        Xsh = Xs.dot(Wxh) + bxh

        if relu_encoders:
            Xsh = np.maximum(Xsh, 0)
            Xi = np.maximum(Xi, 0)

        # recurrence iteration for the Multimodal RNN similar to one described in Karpathy et al.
        d = model['Wd'].shape[0]  # size of hidden layer
        n = Xs.shape[0]
        H = np.zeros((n, d))  # hidden layer representation


        # Hidden Layer 1
        for t in xrange(n):
            #if not rnn_feed_once or t == 0:
                # feed the image in if feedonce is false. And it it is true, then
                # only feed the image in if its the first iteration
            H[t] = np.maximum(Xi + Xsh[t] + bhh, 0)  # also ReLU
          #  else:
          #      H[t] = np.maximum(Xsh[t] + bhh, 0) # also ReLU

        cache = {}

        finalH = H
        for l in range(params.get('layers')):
            L = HLayer(d,None,predict_mode,l)
            finalH, cache,M = L.forward(finalH,n,model,cache)

        if drop_prob_decoder > 0: # if we want dropout on the decoder
            if not predict_mode: # and we are in training mode
                scale2 = 1.0 / (1.0 - drop_prob_decoder)
                U2 = (np.random.rand(*(H.shape)) < (1 - drop_prob_decoder)) * scale2 # generate scaled mask
                H *= U2 # drop!


        # decoder at the end
        Wd = model['Wd']
        bd = model['bd']
        Y = finalH.dot(Wd) + bd

        if not predict_mode:
            # we can expect to do a backward pass
            cache['H'] = finalH
            cache['Wd'] = Wd
            cache['Xs'] = Xs
            cache['Xsh'] = Xsh
            cache['Wxh'] = Wxh
            cache['Xi'] = Xi
            cache['relu_encoders'] = relu_encoders
            cache['drop_prob_encoder'] = drop_prob_encoder
            cache['drop_prob_decoder'] = drop_prob_decoder
            #cache['rnn_feed_once'] = rnn_feed_once
            if drop_prob_encoder > 0:
                cache['Us'] = Us # keep the dropout masks around for backprop
                cache['Ui'] = Ui
            if drop_prob_decoder > 0: cache['U2'] = U2

            cache['layers'] = params.get('layers')
        return Y, cache

    @staticmethod
    def backward(dY, cache):
        '''
        Backwardpropagate the difference dY through the network which is 'saved' in the cache.
        :return: propagated differences
        '''
        Wd = cache['Wd']
        H = cache['H']
        Xs = cache['Xs']
        Xsh = cache['Xsh']
        Wxh = cache['Wxh']
        Xi = cache['Xi']
        layers = cache['layers']
        drop_prob_encoder = cache['drop_prob_encoder']
        drop_prob_decoder = cache['drop_prob_decoder']
        #rnn_feed_once = cache['rnn_feed_once']
        relu_encoders = cache['relu_encoders']
        '''
        rnn_feed_once = cache['rnn_feed_once']
        '''
        n, d = H.shape
        # backprop the decoder
        dWd = H.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        D = dY.dot(Wd.transpose())

        # backprop dropout, if it was applied
        if drop_prob_decoder > 0:
            D*= cache['U2']

        dict = {}

        for l in reversed(range(layers)):

            L = HLayer(d,None,None,l)
            dnew, D = L.backward(cache,D)
            dict.update(dnew)


        dWxh = Xs.transpose().dot(D)
        dbhh = np.sum(D, axis=0, keepdims=True)
        dbxh = np.sum(D, axis=0, keepdims = True)
        dXs = D.dot(Wxh.transpose())
        dXi = np.zeros(d)
        for i in range(n):
            dXi += D[i]

        if relu_encoders:
            # backprop relu
            dXs[Xsh <= 0] = 0
            dXi[Xi <= 0] = 0

        if drop_prob_encoder > 0: # backprop encoder dropout
            dXi *= cache['Ui']
            dXs *= cache['Us']

        dict.update({'bhh': dbhh, 'Wd': dWd, 'bd': dbd, 'Wxh': dWxh, 'bxh': dbxh, 'dXs': dXs, 'dXi': dXi})
        #for key in (dict.keys()):
        #    print(key + ' :' + str(dict[key].shape))

        return dict

    @staticmethod
    def predict(Xi, model, Ws, params, **kwargs):
        '''
        Predicts a sentence based on an image vector Xi and the learned model and other parameters
        '''

        beam_size = kwargs.get('beam_size', 1)
        relu_encoders = params.get('rnn_relu_encoders', 0)
        rnn_feed_once = params.get('rnn_feed_once', 0)

        d = model['Wd'].shape[0]  # size of hidden layer
        bhh = model['bhh']
        Wd = model['Wd']
        bd = model['bd']
        Wxh = model['Wxh']
        bxh = model['bxh']

        if relu_encoders:
            Xi = np.maximum(Xi, 0)

        if beam_size > 1:
            # perform beam search
            # NOTE: code duplication here with lstm_generator
            # ideally the beam search would be abstracted away nicely and would take
            # a TICK function or something, but for now lets save time & copy code around. Sorry ;\
            beams = [(0.0, [], np.zeros(d))]
            nsteps = 0
            while True:
                beam_candidates = []
                for b in beams:
                    ixprev = b[1][-1] if b[1] else 0
                    if ixprev == 0 and b[1]:
                        # this beam predicted end token. Keep in the candidates but don't expand it out any more
                        beam_candidates.append(b)
                        continue
                    # tick the RNN for this beam
                    Xsh = Ws[ixprev].dot(Wxh) + bxh
                    if (not rnn_feed_once) or (not b[1]):
                        if relu_encoders:
                            Xsh = np.maximum(Xsh, 0)

                    h1 = np.maximum(Xi + Xsh  + bhh, 0)
                    '''else:
                        h1 = np.maximum(Xsh + b[2].dot(Whh) + bhh, 0)
                    '''

                    finalH = h1
                    for l in range(params.get('layers')):
                        L = HLayer(d,None,True,l)
                        finalH, cache,M = L.forward(finalH,h1.size()[0],model,{})
                    h1 = finalH
                    y1 = h1.dot(Wd) + bd

                    # compute new candidates that expand out form this beam
                    y1 = y1.ravel()  # make into 1D vector
                    maxy1 = np.amax(y1)
                    e1 = np.exp(y1 - maxy1)  # for numerical stability shift into good numerical range
                    p1 = e1 / np.sum(e1)
                    y1 = np.log(1e-20 + p1)  # and back to log domain
                    top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
                    for i in xrange(beam_size):
                        wordix = top_indices[i]
                        beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1))

                beam_candidates.sort(reverse=True)  # decreasing order
                beams = beam_candidates[:beam_size]  # truncate to get new beams
                nsteps += 1
                if nsteps >= 20:  # bad things are probably happening, break out
                    break
            # strip the intermediates
            predictions = [(b[0], b[1]) for b in beams]

        else:
            ixprev = 0  # start out on start token
            nsteps = 0
            predix = []
            predlogprob = 0.0
            hprev = np.zeros((1, d))  # hidden layer representation
            xsprev = Ws[0]  # start token
            while True:
                Xsh = Ws[ixprev].dot(Wxh) + bxh
                if relu_encoders:
                    Xsh = np.maximum(Xsh, 0)

                #if (not rnn_feed_once) or (nsteps == 0):
                ht = np.maximum(Xi + Xsh  + bhh, 0)
                ''' else:
                    ht = np.maximum(Xsh  + bhh, 0)
                '''
                Y = ht.dot(Wd) + bd
                hprev = ht

                ixprev, ixlogprob = ymax(Y)
                predix.append(ixprev)
                predlogprob += ixlogprob

                nsteps += 1
                if ixprev == 0 or nsteps >= 20:
                    break
            predictions = [(predlogprob, predix)]
        return predictions



def ymax(y):
    """ simple helper function here that takes unnormalized logprobs """
    y1 = y.ravel()  # make sure 1d
    maxy1 = np.amax(y1)
    e1 = np.exp(y1 - maxy1)  # for numerical stability shift into good numerical range
    p1 = e1 / np.sum(e1)
    y1 = np.log(1e-20 + p1)  # guard against zero probabilities just in case
    ix = np.argmax(y1)
    return (ix, y1[ix])
