__author__ = "Wout & Thijs"

import os
import scipy.io
from scipy import spatial
import pickle

from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
from PIL import Image

from Code.imagernn.data_provider import getDataProvider
from Code.cca.stackedCCAModel import *

def stem(word):
    '''
    :param word
    :return: the given word, stemmed using the porter algorithm
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
    :return: a list containing all the words in a hardcoded file
    '''
    result = []
    voc = open('complete_dictionary.txt')
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
    print "amount of words: " + str(len(vocabulary))
    idf = np.zeros(len(vocabulary))
    result = {}
    current = 0
    for dirname, dirnames, filenames in os.walk('Flickr30kEntities/sentence_snippets'):
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
                    for word in line.split():
                        stemmed = stem(word.decode('utf-8')).lower()
                        if stemmed in vocabulary:
                            wordcount += 1
                            i = vocabulary.index(stemmed.lower())
                            row[i] += 1
                    if wordcount:
                        row = row / wordcount
                    for w in range(len(row)):
                        if row[w] > 0:
                            idf[w] += 1
                    if np.linalg.norm(row) > 0 :
                        result[filename[0:-4]+"_"+str(sentenceID)] = row
                line = f.readline()
                sentenceID += 1
    idf = len(result.keys()) / idf
    return result, idf

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

def mainExec(name_file1, name_file2, features1, features2):
    '''
    Given two files with names, and two files with features, perform the Stacked Auxiliary Embedding method
    on two matrices. The first one is the concatenation of both feature lists, the second matrix contains tf-idf weighted
    representations of the training sentences of Flickr30kEntities. The intermediate CCA model is written to disk,
    as well as the final model
    :param name_file1
    :param name_file2
    :param features1
    :param features2
    '''
    print "Creating vocabulary"
    voc = readVocabulary()
    print "Generating document vectors"
    occurrenceVectors, idf = createOccurrenceVectors(voc)
    print "Weighing vectors"
    weightedVectors = weight_tfidf(occurrenceVectors, idf)
    print "creating feature dictionary"
    featuresDict = createFeatDict(weightedVectors.keys(), name_file1, name_file2, features1, features2 )
    imagematrix, sentenceMatrix = createSnippetMatrices(featuresDict, weightedVectors)

    print "Modelling cca"
    cca = CCA(n_components = 128)
    cca = fitCCA(cca, imagematrix, sentenceMatrix, "ccasnippetmodel.p")

    trainingimages, trainingsentences = createTrainMatrices(voc)
    trans_img, trans_sent = cca.transform(trainingimages, trainingsentences)

    nn_img = nearest_neighbor(trainingimages)
    nn_sent = nearest_neighbor(trainingsentences)
    print "NN Image: " + str(nn_img)
    print "NN Sentence: " + str(nn_sent)
    augmented_imgs, augmented_sentences = augmentMatrices(nn_img, nn_sent, trainingimages, trainingsentences, trans_img,
                                                          trans_sent)
    print "Fitting augmented CCA model"
    augmentedcca = CCA(n_components=96)
    augmentedcca = fitCCA(augmentedcca, augmented_imgs, augmented_sentences, "augmentedcca.p")
    print "Writing the model to disk"

    resultingModel = StackedCCAModel(nn_img, nn_sent, cca, augmentedcca)

    pickle.dump(resultingModel, open("completestackedCCAModel.p", 'w+'))


def augmentMatrices(nn_img, nn_sent, trainingimages, trainingsentences, trans_img, trans_sent):
    '''
    :param nn_img: average distance to 50th nearest neighbor of the images
    :param nn_sent: average distance to 50th nearest neighbor of the sentences
    :param trainingimages: matrix containing the training images
    :param trainingsentences: matrix containing the training sentences
    :param trans_img: cca transformation of the training images
    :param trans_sent: cca transformation of the training sentences
    :return: augmented representations of the training images and sentences based on a Random Fourier Feature function.
    '''
    augmented_imgs = []
    augmented_sentences = []
    i = 0
    while i < len(trans_img):
        newSentence = np.append(trainingsentences[i], phi(3000 - len(trainingsentences[i]), nn_sent, trans_sent[i]))
        newImage = np.append(trainingimages[i], phi(3000 - len(trainingimages[i]), nn_img, trans_img[i]))
        if (((not np.linalg.norm(newSentence) == 0) and (np.all(np.isfinite(newSentence))) and (
        not np.any(np.isnan(newSentence)))) and
                ((not np.linalg.norm(newImage) == 0) and (np.all(np.isfinite(newImage))) and (
                not np.any(np.isnan(newImage))))):
            if i % 1000 == 0:
                print "Current pair: " + str(i)
            augmented_imgs.append(newImage)
            augmented_sentences.append(newSentence)
            i += 1
    return np.array(augmented_imgs), np.array(augmented_sentences)


def createSnippetMatrices(featuresDict, weightedVectors):
    '''
    Given a dictionary mapping image names to image features and a dictionary mapping image names to sentence vectors,
    create two matrices containing respectively image features and sentences, in the same order with respect to the
    image name
    :param featuresDict: dictionary containing mapping from image names to image features
    :param weightedVectors: dictionary containing mapping from image names to sentence vectors
    :return: image feature matrix and sentence matrix, in corresponding order
    '''
    sentenceMatrix = []
    imagematrix = []
    print "Creating matrices"
    currentSentence = 0
    for i in weightedVectors.keys():
        currentSentence += 1
        if currentSentence % 1000 == 0:
            print "Current Sentence: " + str(currentSentence)
        imgfeature = featuresDict[i]
        imagematrix.append(imgfeature)
        sentenceMatrix.append(weightedVectors[i])
    return np.array(imagematrix), np.array(sentenceMatrix)


def createTrainMatrices(voc):
    s = getStopwords()
    idf = np.zeros(len(voc))
    trainingimages = []
    trainingsentences = []
    dp = getDataProvider('flickr30k')
    currentPair = 0
    current_image = ""
    current_sentence = []
    for pair in dp.iterImageSentencePair():
        if currentPair % 1000 == 0:
            print "Current pair : " + str(currentPair)
        img_name = pair['image']['filename']
        new_sentence = pair['sentence']['tokens']
        img = pair['image']['feat']
        if(img_name is current_image):
            current_sentence=current_sentence + new_sentence
        else:
            current_image=img_name
            sentence = getFullSentence(current_sentence, voc, s)
            current_sentence = new_sentence
            if np.linalg.norm(sentence) > 0:
                for i in range(len(sentence)):
                    if sentence[i] > 0:
                        idf[i] += 1
                trainingimages.append(img)
                trainingsentences.append(sentence)
                currentPair += 1
    trainingsentences = np.array(trainingsentences)
    trainingimages = np.array(trainingimages)
    for i in range(len(trainingsentences)):
        trainingsentences[i] = trainingsentences[i] * idf
    return trainingimages, trainingsentences


def fitCCA(model, x, y, file):
    '''
    fit the given model on the given date, write it to disk and return it
    :param model
    :param x: first part of dataset
    :param y: second part of dataset, corresponding to x
    :param file: file to write to
    :return: fitted model
    '''
    model.fit(x, y)
    pickle.dump(model, open(file, 'w+'))
    return model


def createFeatDict(names, namesfile1, namesfile2, features1, features2):
    '''
    Given a list of names, two files containing names and two lists of features, return a dictionary
    containing all the names in the given list, mapping them to the correct features
    :param names: list of names
    :param namesfile1: first file containing names
    :param namesfile2: second file containing names
    :param features1: first list of features
    :param features2: second list of features
    :return: dictionary containing all the names in the given list, mapping them to the correct features
    '''
    result = {}
    current = 0
    for name in names:
        current += 1
        if current % 1000 == 0:
            print "current image: "+ str(current)
            img1 = getImage(name, namesfile1, features1)
            if not len(img1) == 0:
                result[name] = img1
            else:
                result[name] = getImage(name, namesfile2, features2)
    return result

def phi(wantedDimension, sigma, x):
    '''
    :param wantedDimension
    :param sigma: parameter of the RFF
    :param x: vector to modify
    :return: RFF function applied to the given vector, based on the given sigma and wanted dimension
    '''
    b = np.random.rand(wantedDimension)
    R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
    return np.dot(x,R) + b

def nearest_neighbor(matrix):
    '''
    :param matrix
    :return: average distance to the 50th nearest neighbor of all observations in the given matrix
    '''
    avg_dist = 0
    for i in range(len(matrix)):
        x = np.linalg.norm(matrix[i])
        distances = np.zeros(len(matrix))
        for j in range(len(matrix)):
            if not i == j:
                distances[j] = spatial.distance.cosine(matrix[i]/np.linalg.norm(matrix[i]), matrix[j]/np.linalg.norm(matrix[j]))
        distances = np.delete(distances, i)
        distances.sort()
        avg_dist += distances[49]
    
    avg_dist = avg_dist / len(matrix)
    return avg_dist


def getFullSentence(sentence, vocabulary, stopwords):
    '''
    :param sentence: sentence to count the words of
    :param vocabulary: vocabulary to use
    :param stopwords: stopwords to remove from the sentence
    :return: array containing the counts of each word in the vocabulary
    '''
    vector =  np.zeros(len(vocabulary))
    result = remove_common_words(sentence, stopwords)
    for word in result :
        if word.lower() in vocabulary :
            vector[vocabulary.index(word.lower())] += 1
    return vector

def remove_common_words(sentence,stopwords):
    '''
    :param sentence
    :param stopwords
    :return: copy of the sentence, stripped of words that are in the provided stopwords and words with length <3
    '''
    stopwords.add(' ') #add spaces to stopwords
    result = []
    for word in sentence:
        if not word.lower() in stopwords and len(word)>2:
            result.append(word.lower())
    return result

def isLargeEnough(filename):
    '''
    :param filename: image to check
    :return: True if the image behind the filename is bigger than 64x64
    '''
    file = filename+".jpg"
    try:
        image = Image.open("../Flickr30kEntities/image_snippets/"+file)
    except IOError:
        # image not found. Is ok, many snippets dont have a corresponding image
        return False
    width, height = image.size
    return (width >=64 ) and (height >= 64)

def getImage(filename, file_with_names, features):
    '''
    :param filename: like 123456789_1_1
    :param file_with_names: file with image names
    :param features: list of image features
    :return: the feature corresponding to the given filename
    '''
    file_with_names = open(file_with_names)
    line = file_with_names.readline()
    linenumber = 0
    while(not line == ""):
        if line[0:-5] == filename: # remove last 4 characters of the name (_x_x)
            return features[linenumber]
        line = file_with_names.readline()
        linenumber+=1
    return []

if __name__ == "__main__":
    # image features are split into two matrices, because they were too large to compute
    names1 = "../Flickr30kEntities/image_snippets/images.txt"
    names2 = "../Flickr30kEntities/image_snippets/images2.txt"
    feats1 = scipy.io.loadmat("../Flickr30kEntities/snippets_features/vgg_feats.mat")['feats'].transpose()
    feats2 = scipy.io.loadmat("../Flickr30kEntities/snippets_features/vgg_feats2.mat")['feats'].transpose()
    mainExec(names1, names2 , feats1, feats2)


