__author__ = 'Thijs'

import argparse
from lda_images.topic_extractor import TopicExtractor
import numpy

def main(params):
    iterations = params['iterations']
    dataset = params['dataset']
    topics = params['topics']
    topic_extractor = TopicExtractor(dataset,topics,iterations)
    model,vocabulary,images = topic_extractor.extract_model()
    test_topic(model,vocabulary)
    save_image_topic_distribution(model,images,dataset,topics)
    print('finished')
def test_topic(model,vocabulary):
    n = 10
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocabulary)[numpy.argsort(topic_dist)][:-(n+1):-1]
        print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

def save_image_topic_distribution(model,images,dataset,topics):
    f = open('lda_images/image_topic_distribution_'+dataset+'top'+str(topics)+'.txt','w')
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

