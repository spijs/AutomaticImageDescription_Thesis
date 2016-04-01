__author__ = 'Wout & thijs'

import argparse
#from lda_images.lda_learner import *
from lda_images.new_lda_learner import *

# given a dataset and an amount of topics, train a neural network
# to map image representations onto topic distribtutions
def main(params):
    dataset = params['dataset']
    topics = params['topics']
    rate = params['rate']
    iterations = params['iterations']
    hidden_layers = params['hidden']
    layers = params['layers']
    pert = params['pert']
    networkLearner = LDANetworkLearner(dataset, topics, rate, hidden_layers,layers, pert)
    networkLearner.learnNetwork(iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr30k', help='dataset: flickr8k/flickr30k')
    parser.add_argument('-t', '--topics', dest='topics', type=int, default=120, help='Number of topics to learn lda model')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default= 1000000, help='Number of iterations for training the network')
    parser.add_argument('-r', '--rate', dest='rate', type=float, default=0.001, help='Training rate for the neural network')
    parser.add_argument('-hidden', '--hidden', dest='hidden', type=int, default=256, help='Number of hidden neurons per layer')
    parser.add_argument('-l', '--layers', dest='layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-pert', '--pert', dest='pert', type=int, default=0, help="=0 if you dont want to use perturbed dataset")
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)
