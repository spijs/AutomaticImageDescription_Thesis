import os
import numpy as np
from nltk.stem.porter import *
from sklearn.cross_decomposition import CCA
import scipy.io
import pickle
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
    result = {}
    current = 0
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
	for filename in filenames:
            current += 1
            print "current sentence : " + str(current)
            f= open('./Flickr30kEntities/sentence_snippets/'+filename)
            line = f.readline()
            sentenceID = 1
            while not (line == ""):
                wordcount = 0
                row = np.zeros(len(vocabulary))
                for word in line:
                    stemmed = stem(word.decode('utf-8'))
                    if stemmed in vocabulary:
                        wordcount += 1
                        i = vocabulary.index(stemmed)
                        row[i] += 1
                if wordcount:
                    row = row / wordcount
                result[filename+"_"+str(sentenceID)] = row
                line = f.readline()
                sentenceID += 1
    return result

def get_idf(documents, vocabulary):
    result = np.zeros(len(vocabulary))
    for i in range(len(vocabulary)):
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

def mainExec(name_file, features):
    voc = createVocabulary()
    occurrenceVectors = createOccurrenceVectors(voc)
    idf = get_idf(occurrenceVectors, voc)
    weightedVectors = weight_tfidf(occurrenceVectors, idf, voc)

    sentenceMatrix = []
    imagematrix = []
    for i in weightedVectors.keys():
        sentenceMatrix = sentenceMatrix.append(weightedVectors[i])
        imagematrix = imagematrix.append(getImage(i,name_file, features))
    cca = CCA(n_components=128)
    cca.fit(sentenceMatrix, imagematrix)
    pickle.dump(cca, open("ccasnippetmodel.p",'w+'))

    dp = getDataProvider('flickr30k')
    for pair in dp.sampleImageSentencePair():
        img = pair['image']['feat']
        sentence = getFullSentence(pair)


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


