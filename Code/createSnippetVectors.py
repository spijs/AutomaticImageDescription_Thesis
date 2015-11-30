import os
import numpy as np
from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
import scipy.io
from scipy import spatial
import pickle
from PIL import Image
from imagernn.data_provider import getDataProvider


''' stems a word by using the porter algorithm'''
def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word)

def getStopwords():
        stopwords = set()
        file=open('lda_images/english')
        for line in file.readlines():
            stopwords.add(line[:-1])
        return stopwords

'''
Creates a vocabulary, being a list of words
'''
def createVocabulary():
    dict = {}
    result = {}
    stopwords = getStopwords()
    current = 0
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
	    current += 1
	    print "Preprocessing sentence: " + str(current)
            f= open('./Flickr30kEntities/sentence_snippets/'+filename)
            line = f.readline()
            # print filename
            while not (line == ""):
                for word in line.split():
                    word = stem(word.decode('utf-8'))
                    if (not word in stopwords):
                        if(not word in dict):
                            dict[word]=1
                        else:
                            dict[word]+=1
                line = f.readline()
        for word in dict:
            if(dict[word] >= 5):
                result[word]=dict[word]
    return result.keys()

'''
Reads a set of documents, returns a dictionary containing the filename of the corresponding picture, and the
unweighted bag of words representation of the sentences in the documents, based on the given vocabulary
'''
def createOccurrenceVectors(vocabulary):
    idf = np.zeros(len(vocabulary))
    result = {}
    current = 0
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
	for filename in filenames:
        current += 1
        if current % 1000 == 0:
		    print "current sentence : " + str(current)
        f= open('./Flickr30kEntities/sentence_snippets/'+filename)
        line = f.readline()
        sentenceID = 1
        while not (line == ""):
            wordcount = 0
            row = np.zeros(len(vocabulary))
            for word in line.split():
                stemmed = ""
                try:
                    stemmed = stem(word.decode('utf-8'))
                except UnicodeDecodeError:
                     print "This word gave an error: " + word
                if stemmed in vocabulary:
                    wordcount += 1
                    i = vocabulary.index(stemmed)
                    row[i] += 1
            if wordcount:
                row = row / wordcount
            for w in range(len(row)):
                if row[w] > 0:
                    idf[w] += 1
            result[filename+"_"+str(sentenceID)] = row
            line = f.readline()
            sentenceID += 1
    idf = len(result.keys()) / idf
    return result, idf

def get_idf(documents, vocabulary):
    result = np.zeros(len(vocabulary))
    current = 0
    for i in range(len(vocabulary)):
        current += 1
	if current % 100 == 0:
	    print "Current word: " + str(current)
        empty = True
        docCount = 0
        for j in documents.keys():
            if documents[j][i] > 0:
                empty = False
                docCount += 1
        if not empty:
            result[i] = docCount
    result = len(documents.keys()) / result
    return result

def weight_tfidf(documents, inv_freq, vocabulary):
    result = {}
    for i in documents.keys():
        doc = documents[i]
        result[i] = doc * inv_freq
    return result

def mainExec(name_file, features):
    print "Creating vocabulary"
    voc = createVocabulary()
    print "Generating document vectors"
    occurrenceVectors, idf = createOccurrenceVectors(voc)
    # print "Generating idf weights"
    # idf = get_idf(occurrenceVectors, voc)
    print "Weighing vectors"
    weightedVectors = weight_tfidf(occurrenceVectors, idf, voc)

    sentenceMatrix = []
    imagematrix = []
    print "Creating matrices"
    for i in weightedVectors.keys():
        if isLargeEnough(i):
            sentenceMatrix = sentenceMatrix.append(weightedVectors[i])
            imagematrix = imagematrix.append(getImage(i,name_file, features))
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
        sentence = getFullSentence(pair, voc)
        for i in range(len(sentence)):
            if sentence[i] > 0:
                idf[i] += 1
        trainingsentences.append(sentence)
    for i in range(trainingsentences):
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
    augmentedcca.fit(augm_img, augm_sent)

    pickle.dump(cca, open("augmentedcca.p",'w+'))



def phi(wantedDimension, sigma, x):
    b = np.random.rand(wantedDimension)
    R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
    return np.dot(x,R) + b


def nearest_neighbor(matrix):
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
    sentences = imagesentencepair['image']['sentences']
    s = getStopwords()
    full = []
    for sentence in sentences:
        result = remove_common_words(sentence['tokens'], s)
        full.extend(result)

def remove_common_words(sentence,stopwords):
        #s = set(stopwords.words('english'))
        stopwords.add(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word.lower() in stopwords and len(word)>2:
                result.append(word.lower())
        return result

def isLargeEnough(filename):
    file = filename+".jpg"
    image = Image.open("./Flickr30kEntities/image_snippets/"+file)
    width, height = image.size
    return (width >= 64) and (height >= 64)

def getImage(filename, file_with_names, features):
    line = file_with_names.readline()
    linenumber = 0
    while(not line == ""):
        if line+".jpg" == filename:
            return features[linenumber]
        line = file_with_names.readline()
        linenumber+=1


if __name__ == "__main__":
    names = open("./Flickr30kEntities/image_snippets/images.txt")
    feats = scipy.io.loadmat("./Flickr30kEntities/image_snippets/vgg_feats.mat")['feats'].transpose()
    mainExec(names, feats)


