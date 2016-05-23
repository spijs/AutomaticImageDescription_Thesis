__author__ = 'Wout & Thijs'

import numpy as np

def main():
    d = create_dist_dict('models/image_topic_distribution_flickr30k_top120_test_pert.txt')
    topics = createTopicList(120)
    evaluate(d,topics)

def evaluate(dict,topics):
    '''
    Write the five most probable topic names and their probabilities to a file.
    :param dict: dictionary containing image names and probability distributions over all topics
    :param topics: names of the topics
    '''
    file = open('models/highest_topics_test_120_pert.txt','w')
    for key in dict.keys():
        dist = dict[key]
        indices = get_n_highest_indices(dist,5)
        file.write('Image %s topics: \n' % key)
        for i in range(5):
            file.write('Topic %i: %s with probability %f\n' % (i,topics[indices[i]].rstrip(),dist[indices[i]]))
    file.close()

def get_n_highest_indices(list,n):
    '''
    :param list: The list to take the highest values from
    :param n: number of values to return
    :return: The indices of the n highest values in the given list
    '''
    arr = np.array(list)
    arr = arr.argsort()[-n:][::-1]
    return np.asarray(arr)

def create_dist_dict(filename):
    '''
    :param filename: file to read
    :return: a dictionary mapping imagenames to topic distributions, read from the given file
    '''
    dict = {}
    f = open(filename)
    rawDist = []
    line = f.readline()
    while(line != ''):
        split = line.split()
        if '[' in split and len(rawDist)!= 0:
            img, distribution = preprocess(rawDist)
            dict[img] = distribution
            rawDist = split
        else:
            rawDist.extend(split)
        line = f.readline()
    img, distribution = preprocess(rawDist)
    dict[img] = distribution
    return dict

def preprocess(rawDistribution):
    '''
    Given a list, containing both image name and topic distribution, extract the name and distribution and return
    :param rawDistribution: list to process
    :return: extracted image name and distribution
    '''
    imgname = rawDistribution[0]
    distribution = []
    for i in range(2,len(rawDistribution)):
        modifiedNumber = str(rawDistribution[i]).replace(']', '')
        if modifiedNumber!= '':
            m = float(modifiedNumber)
            distribution.extend([m])
    return imgname, distribution


def createTopicList(nbOfTopics=120):
    '''
    create a list containing the 10 most probable words for the given number of topics. For now, the parameter is not
    used and the filename is hardcoded.
    :param nbOfTopics: the number of topics to use
    :return: list containing the 10 most probable topics for LDA model with given number of topics
    '''
    file = open('models/topic_word_distribution_flickr30ktop120.txt', 'r') # hardcoded. change for different topic model
    list = []
    l = file.readline()
    l = file.readline()
    while l != "":
        list.extend(l.split('*')[0][2:])
    return list

if __name__ == "__main__":
    main()
