__author__ = 'Wout & thijs'

import argparse
from lda_images.topic_extractor import TopicExtractor
import numpy

''' This class is used to create topic distributions for each split in a given dataset based on a model
    learned with only the training split. '''
def main(params):
    iterations = params['iterations']
    dataset = params['dataset']
    pert = params['pert']
    topics = params['topics']
    topic_extractor = TopicExtractor(dataset,topics,iterations, pert)
    model,vocabulary,splitPairs = topic_extractor.extract_model()
    test_topic(model,vocabulary,dataset,topics, pert)
    save_image_topic_distribution(model,splitPairs['train'].keys(),dataset,topics, pert)

    # Learn the 'ground truth' lda topic distributions for test and validation split.
    for split in ['test', 'val']:
        sentences = splitPairs[split].values()
        matrix = topic_extractor.create_document_term_matrix(sentences, vocabulary)
        predict_image_topic_distribtution(model, splitPairs, dataset, topics, split, matrix, pert)
    print('finished')

''' Learn and output the topic distribution of a given split in a given lda model'''
def predict_image_topic_distribtution(model, image_sentence_pairs, dataset, topics, split, matrix, pert = None):
    pert_string = ''
    if pert:
        pert_string = "_pert_"
    pairs = image_sentence_pairs[split]
    doc_topic = model.transform(matrix)
    f= open('lda_images/models/image_topic_distribution_'+dataset+'top'+str(topics)+'_'+split+pert_string+'.txt','w')
    imagenames = pairs.keys()
    numpy.set_printoptions(suppress=True)
    save_distribtutions(doc_topic, f, imagenames)

''' Write a topic distribution matrix to the given file, with the given file names'''
def save_distribtutions(distribution_matrix, file, imagenames):
    for n in range(len(distribution_matrix)):
        dist = distribution_matrix[n, :]
        im = imagenames[n]
        file.write(im + ' ' + str(dist) + '\n')


''' This method is used to write the learned topic distribution to a file'''
def save_image_topic_distribution(model,images,dataset,topics, pert = None):
    pert_string = ''
    if pert:
        pert_string = "_pert_"
    doc_topic = model.doc_topic_
    f = open('lda_images/models/image_topic_distribution_'+dataset+'top'+str(topics)+'_train'+pert_string+'.txt','w')
    numpy.set_printoptions(suppress=True)
    save_distribtutions(doc_topic, f, images)

''' This method is used to output the learned topic and for each topic the most important words'''
def test_topic(model,vocabulary,dataset,topics, pert = None):
    pert_string = ''
    if pert:
        pert_string = "_pert_"
    f= open('lda_images/models/topic_word_distribution_'+dataset+'top'+str(topics)+pert_string+'.txt','w')
    n = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n+1):-1]
        f.write('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))





''' Parses the given arguments'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default=0, help='Number of iterations to learn lda model')
    # global setup settings, and checkpoints
    parser.add_argument('-d', '--dataset', dest='dataset', default='flickr8k', help='dataset: flickr8k/flickr30k')
    parser.add_argument('-pert', '--perturbated', dest='pert', default=False, help='Whether or not to use the perturbed dataset')
    parser.add_argument('-t', '--topics', dest='topics', type=int, default=80, help='Number of topics to learn lda model')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)

