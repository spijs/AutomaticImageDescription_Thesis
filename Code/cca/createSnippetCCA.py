__author__ = "Wout & Thijs"

import os
import numpy as np
from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
import scipy.io
from scipy import spatial
import pickle
from PIL import Image
from Code.imagernn.data_provider import getDataProvider


def stem(word):
    '''
    :param word
    :return: given word, stemmed using a Porter algorithm
    '''
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def getStopwords():
    '''
    :return: a list containing the most frequent english words
    '''
    stopwords = set()
    file=open('../lda_images/english')
    for line in file.readlines():
        stopwords.add(line[:-1])
    return stopwords

def readVocabulary():
    '''
    Reads a vocabulary from an hardcoded file
    '''
    result = []
    voc = open('dictionary.txt')
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result

def createOccurrenceVectors(vocabulary):
    '''
    Reads a set of documents, returns a dictionary containing the filename of the corresponding picture, and the
    unweighted bag of words representation of the sentences in the documents, based on the given vocabulary
    :param vocabulary
    :return: dictionary mapping filenames to unweighted bag-of-word representation, together with the idf weights
    for the given vocabulary
    '''
    idf = np.zeros(len(vocabulary))
    result = {}
    current = 0
    for dirname, dirnames, filenames in os.walk('../Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
            current += 1
            if current % 1000 == 0:
                print "current sentence : " + str(current)
            f = open('../Flickr30kEntities/sentence_snippets/'+filename)
            line = f.readline()
            sentenceID = 1
            while not (line == ""):
                wordcount = 0
                if isLargeEnough(filename[0:-4]+"_"+str(sentenceID)):
                    row = np.zeros(len(vocabulary))
                    for word in line.split(): # loop over all words in the read sentence
                        stemmed = stem(word.decode('utf-8')).lower()
                        if stemmed in vocabulary:
                            wordcount += 1
                            i = vocabulary.index(stemmed.lower())
                            row[i] += 1 # add one to the value corresponding to the stemmed word
                    if wordcount: # if there are any used words in the sentence
                        row = row / wordcount
                    for w in range(len(row)):
                        if row[w] > 0:
                            idf[w] += 1 # add one to the corresponding idf weight
                    result[filename[0:-4]+"_"+str(sentenceID)] = row
                line = f.readline()
                sentenceID += 1
    idf = len(result.keys()) / idf # calculate idf weights
    return result, idf

'''

'''
def weight_tfidf(documents, inv_freq):
    '''
    :param documents
    :param inv_freq: inverse document frequencies for the set of documents
    :return: weighted version of each document with the given idf vector
    '''
    result = {}
    for i in documents.keys():
        doc = documents[i]
        result[i] = doc * inv_freq
    return result

def mainExec(name_file, features):
    '''
    Based on a list of image names and image features, learn a CCA model based on Stacked Auxiliary Embedding and
    save this model to disk.
    :param name_file
    :param features
    :return:
    '''
    print "Creating vocabulary"
    voc = readVocabulary()
    print "Generating document vectors"
    occurrenceVectors, idf = createOccurrenceVectors(voc)
    print "Weighing vectors"
    weightedVectors = weight_tfidf(occurrenceVectors, idf)

    sentenceMatrix = []
    imagematrix = []
    print "Creating matrices"
    currentSentence = 0
    for i in weightedVectors.keys():
        if isLargeEnough(i):
            currentSentence += 1
            print "current Sentence: " + str(currentSentence)
            for j in range(len(weightedVectors[i])):
                weightedVectors[i][j] = float(weightedVectors[i][j])
            if currentSentence == 1:
                sentenceMatrix = weightedVectors[i]
                imagematrix = getImage(i,name_file, features)
            elif currentSentence ==2:
                sentenceMatrix = np.concatenate(([sentenceMatrix], [weightedVectors[i]]), axis = 0)
                imagematrix = np.concatenate(([imagematrix], [getImage(i,name_file, features)]), axis = 0)
            else:
                sentenceMatrix = np.concatenate((sentenceMatrix, [weightedVectors[i]]), axis = 0)
                imagematrix = np.concatenate((imagematrix, [getImage(i,name_file, features)]), axis = 0)

    print "Modelling cca"
    cca = CCA(n_components=128)
    cca.fit(sentenceMatrix, imagematrix)
    pickle.dump(cca, open("ccasnippetmodel.p",'w+'))

    idf = np.zeros(len(voc))
    trainingimages = []
    trainingsentences = []
    dp = getDataProvider('flickr30k')
    currentPair = 0
    for pair in dp.sampleImageSentencePair():
        currentPair += 1
        if currentPair % 100 == 0:
            print "Current pair: " + str(currentPair)
        img = pair['image']['feat']
        trainingimages.append(img)
        sentence = getFullSentence(pair)
        for i in range(len(sentence)):
            if sentence[i] > 0:
                idf[i] += 1
        trainingsentences.append(sentence)
    for i in range(len(trainingsentences)):
        trainingsentences[i] = trainingsentences[i]*idf

    trans_img, trans_sent = cca.transform(trainingimages, trainingsentences)
    nn_img = nearest_neighbor(trainingimages)
    nn_sent = nearest_neighbor(trainingsentences)

    augmented_imgs = []
    augmented_sentences = []
    for i in range(len(trans_img)):
        augm_img = trainingimages[i].extend(phi(3000,nn_img, trans_img[i]))
        augmented_imgs.append(augm_img)

    for i in range(len(trans_sent)):
        augm_sent = trainingsentences[i].extend(phi(3000, nn_sent, trans_sent[i]))
        augmented_sentences.append(augm_sent)

    augmentedcca = CCA(n_components= 96)
    augmentedcca.fit(augmented_sentences, augmented_imgs)

    pickle.dump(cca, open("augmentedcca.p",'w+'))

def phi(wantedDimension, sigma, x):
    '''
    :param wantedDimension
    :param sigma
    :param x:  vector to apply the function to
    :return: RFF function of the given vector, based on the given sigma and wanted dimension
    '''
    b = np.random.rand(wantedDimension)
    R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
    return np.dot(x,R) + b


def nearest_neighbor(matrix):
    '''
    :param matrix: matrix with an observation on each row
    :return: the average distance to the 50th nearest neighbor
    '''
    avg_dist = 0
    for i in range(len(matrix)):
        distances = np.zeros(len(matrix)-1)
        for j in range(len(matrix)):
            if not i == j:
                distances[j] = spatial.distance.cosine(matrix[i], matrix[j])
        distances = distances.sort()
        avg_dist += distances[49]
    avg_dist = avg_dist / len(matrix)
    return avg_dist


def getFullSentence(imagesentencepair):
    '''
    :param imagesentencepair: pair of an image with 5 sentences
    :return: array containing the concatenation of the 5 sentences in the pair
    '''
    sentences = imagesentencepair['image']['sentences']
    s = getStopwords()
    full = []
    for sentence in sentences:
        result = remove_common_words(sentence['tokens'], s)
        full.extend(result)
    return full

def remove_common_words(sentence,stopwords):
    '''
    Remove the stopwords and words shorter than 3 characters from the given sentence
    :param sentence
    :param stopwords
    :return: sentence with stopwords and words shorter than 3 characters removed
    '''
    stopwords.add(' ') #add spaces to stopwords
    result = []
    for word in sentence:
        if not word.lower() in stopwords and len(word)>2:
            result.append(word.lower())
    return result

def isLargeEnough(filename):
    '''
    Given a filename, checks if the image behind that filename is bigger than 64x64
    :param filename:
    :return: True if the file is larger than 64x64 pixels
    '''
    file = filename+".jpg"
    try:
        image = Image.open("../Flickr30kEntities/image_snippets/"+file)
    except IOError:
        return False
    width, height = image.size
    return (width >= 64) and (height >= 64)

def getImage(filename, file_with_names, features):
    '''
    :param filename: filename of the wanted feature
    :param file_with_names: file with image names
    :param features: list of image features
    :return: image feature corresponding to the provided image name
    '''
    line = file_with_names.readline()
    linenumber = 0
    while(not line == ""):
        if line+".jpg" == filename:
            return features[linenumber]
        line = file_with_names.readline()
        linenumber+=1


if __name__ == "__main__":
    names = open("../Flickr30kEntities/image_snippets/images.txt")
    feats = scipy.io.loadmat("../Flickr30kEntities/image_snippets/vgg_feats.mat")['feats'].transpose()
    mainExec(names, feats)


