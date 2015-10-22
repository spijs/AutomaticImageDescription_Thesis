__author__ = 'Thijs'

import argparse
from lda_images.topic_extractor import TopicExtractor
import numpy

def main(params):
    iterations = params['iterations']
    dataset = params['dataset']
    topics = params['topics']
    topic_extractor = TopicExtractor(dataset,topics,iterations)
    model,vocabulary = topic_extractor.extract_model()
    test_topic(model,vocabulary)
    print('finished')

def test_topic(model,vocabulary):
    n = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default=0, help='Number of iterations to learn lda model')
    # global setup settings, and checkpoints
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
    parser.add_argument('-t', '--topics', dest='topics', type=int, default=80, help='Number of topics to learn lda model')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)

