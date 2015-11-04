__author__ = 'Thijs'

import argparse
from lda_images.topic_extractor import TopicExtractor
import numpy

def main(params):
    iterations = params['iterations']
    dataset = params['dataset']
    topics = params['topics']
    topic_extractor = TopicExtractor(dataset,topics,iterations)
    model,vocabulary,splitPairs = topic_extractor.extract_model()
    test_topic(model,vocabulary,dataset,topics)
    save_image_topic_distribution(model,splitPairs['train'].keys(),dataset,topics)
    for split in ['test', 'val']:
        sentences = splitPairs[split].values()
        matrix = topic_extractor.create_document_term_matrix(sentences, vocabulary)
        predict_image_topic_distribtution(model, splitPairs, dataset, topics, split, matrix)
    print('finished')

def predict_image_topic_distribtution(model, image_sentence_pairs, dataset, topics, split, matrix):
    pairs = image_sentence_pairs[split]
    f= open('lda_images/models/topic_word_distribution_'+dataset+'top'+str(topics)+'_'+split+'.txt','w')
    doc_topic = model.transform(matrix)
    for n in range(len(doc_topic)):
        dist = doc_topic[n,:]
        im = pairs.keys()[n]
        f.write(im+' '+str(dist) + '\n')


def test_topic(model,vocabulary,dataset,topics):
    f= open('lda_images/models/topic_word_distribution_'+dataset+'top'+str(topics)+'.txt','w')
    n = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n+1):-1]
        f.write('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

def save_image_topic_distribution(model,images,dataset,topics):
    f = open('lda_images/models/image_topic_distribution_'+dataset+'top'+str(topics)+'.txt','w')
    doc_topic = model.doc_topic_
    for n in range(len(doc_topic)):
        dist = doc_topic[n,:]
        im = images[n]
        f.write(im+' '+str(dist) + '\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default=0, help='Number of iterations to learn lda model')
    # global setup settings, and checkpoints
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
    parser.add_argument('-t', '--topics', dest='topics', type=int, default=80, help='Number of topics to learn lda model')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)

