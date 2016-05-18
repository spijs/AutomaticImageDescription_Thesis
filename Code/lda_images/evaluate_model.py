__author__ = 'spijs'

import numpy as np

def main():
    d = create_dist_dict('models/image_topic_distribution_flickr30k_top120_test.txt')
    topics = createTopicList(120)
    evaluate(d,topics)

def evaluate(dict,topics):
    file = open('models/highest_topics_test_120_pert.txt','w')
    for key in dict.keys():
        dist = dict[key]
        indices = get_n_highest_indices(dist,5)
        file.write('Image %s topics: \n' % key)
        for i in range(5):
            file.write('Topic %i: %s with probability %f\n' % (i,topics[indices[i]].rstrip(),dist[indices[i]]))
    file.close()

def get_n_highest_indices(list,n):
    arr = np.array(list)
    arr = arr.argsort()[-n:][::-1]
    return np.asarray(arr)

def create_dist_dict(filename):
    dict = {}
    f = open(filename)
    rawDist = []
    line = f.readline()
    while(line != ''):
        # print 'LINE', line
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
    imgname = rawDistribution[0]
    distribution = []
    for i in range(2,len(rawDistribution)):
        modifiedNumber = str(rawDistribution[i]).replace(']', '')
        # print modifiedNumber
        if modifiedNumber!= '':
            m = float(modifiedNumber)
            distribution.extend([m])
    return imgname, distribution


def createTopicList(nbOfTopics=120):
    for i in range(nbOfTopics):
        list.extend([str(i)])
    return list

if __name__ == "__main__":
    main()