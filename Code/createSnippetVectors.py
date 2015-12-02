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

'''Returns a list containing the most frequent english words'''
def getStopwords():
        stopwords = set()
        file=open('lda_images/english')
        for line in file.readlines():
            stopwords.add(line[:-1])
        return stopwords

'''
Creates a vocabulary based on a folder. Returns a list of words
'''
def readVocabulary():
    result = []
    voc = open('fullDictionary.txt')
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result

'''
Reads a set of documents, returns a dictionary containing the filename of the corresponding picture, and the
unweighted bag of words representation of the sentences in the documents, based on the given vocabulary
'''
def createOccurrenceVectors(vocabulary):
    print "amount of words: " + str(len(vocabulary))
    idf = np.zeros(len(vocabulary))
    result = {}
    current = 0
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
         # if current < 1000:
            current += 1
            if current % 1000 == 0:
                print "current sentence : " + str(current)
            f = open('./Flickr30kEntities/sentence_snippets/'+filename)
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
                    result[filename[0:-4]+"_"+str(sentenceID)] = row
                line = f.readline()
                sentenceID += 1
                #print "ROW: " + str(row)
    for item in idf:
      if item <= 0:
        print "Idf item: " + str(item)
    idf = len(result.keys()) / idf
    return result, idf

'''
Given a set of document vectors, and an inverse document frequency vector, returns the multiplication
of each document with the idf vector
'''
def weight_tfidf(documents, inv_freq):
    result = {}
    for i in documents.keys():
        doc = documents[i]
        result[i] = doc * inv_freq
        #print "Is NaN"+np.any(np.isnan(result[i]))
        #print "infinity "+np.any(np.isinf(result[i]))
    return result

def mainExec(name_file, features):
    print "Creating vocabulary"
    voc = readVocabulary()
    print "Generating document vectors"
    occurrenceVectors, idf = createOccurrenceVectors(voc)
    # print "Generating idf weights"
    # idf = get_idf(occurrenceVectors, voc)
    print "Weighing vectors"
    weightedVectors = weight_tfidf(occurrenceVectors, idf)
    print "creating feature dictionary"
    featuresDict = createFeatDict(weightedVectors.keys(), name_file, features )
    sentenceMatrix = []
    imagematrix = []
    print "Creating matrices"
    currentSentence = 0
    for i in weightedVectors.keys():
    # print sentenceMatrix
        if isLargeEnough(i):
            currentSentence += 1
            #print "Trying stuff for file with name : "+i
#	    print "TRUE"
            for j in range(len(weightedVectors[i])):
                weightedVectors[i][j] = float(weightedVectors[i][j])
            imgfeature = featuresDict[i]
            for j in range(len(imgfeature)):
                imgfeature[j] = float(imgfeature[j])
            #print str(len(weightedVectors[i]))
            if currentSentence == 1:
                sentenceMatrix = weightedVectors[i]
                imagematrix = imgfeature
            elif currentSentence ==2:
                sentenceMatrix = np.append([sentenceMatrix], [weightedVectors[i]], axis = 0)
                imagematrix = np.concatenate(([imagematrix], [imgfeature]), axis = 0)
            else:
                sentenceMatrix = np.append(sentenceMatrix, [weightedVectors[i]], axis = 0)
                imagematrix = np.concatenate((imagematrix, [imgfeature]), axis = 0)
            # imagematrix.append(getImage(i,name_file, features))
#	else: 
	    #print "FALSE"
    # if (sentenceMatrix.dtype.char in np.typecodes['AllFloat']):
    #     print "Type Code is in AllFloat"
    # if not np.isfinite(sentenceMatrix.sum()):
    #     print "Sum of matrix is not finite"
    # if not np.isfinite(sentenceMatrix).all():
    #     print "Not all items are finite"
    #sentenceMatrix = sentenceMatrix
    #imagematrix = imagematrix
    #sentenceMatrix = imagematrix
    print "Sentences: " + str(sentenceMatrix.shape)
    print "Images: " + str(imagematrix.shape)
    #print type(sentenceMatrix)
    #imagematrix = sentenceMatrix
    #sentenceMatrix = imagematrix
    #print imagematrix
    #print "Is Sum finite? : " + str(np.isfinite(sentenceMatrix.sum()))
    #print "ALl items finite? :" + str(np.isfinite(sentenceMatrix).all())
    #print "Amount of samples :" + str(len(sentenceMatrix))
    #pickle.dump(sentenceMatrix, open("sentences.p",'w+'))
    #pickle.dump(imagematrix, open("imagemat.p", 'w+'))
    print "Modelling cca"
    cca = CCA(n_components = 10)
    cca.fit(sentenceMatrix, imagematrix)
    pickle.dump(cca, open("ccasnippetmodel.p",'w+'))

    idf = np.zeros(len(voc))
    trainingimages = []
    trainingsentences = []
    dp = getDataProvider('flickr30k')
    currentPair = 0
    for pair in dp.iterImageSentencePair(max_images= 5):
        currentPair += 1
        img = pair['image']['feat']
        # trainingimages.append(img)
        sentence = getFullSentence(pair)
        for i in range(len(sentence)):
            if sentence[i] > 0:
                idf[i] += 1
            if currentPair == 1:
                trainingsentences = sentence
                trainingimages = img
            elif currentPair ==2:
		print "shape of matrix"+ str(trainingsentences.shape)
	        print "shape of sentence" + str([sentence].shape)
                trainingsentences = np.append([trainingsentences], [sentence], axis = 0)
                trainingimages = np.concatenate(([trainingimages], [img]), axis = 0)
            else:
                trainingsentences = np.append(trainingsentences, [sentence], axis = 0)
                trainingimages = np.concatenate((trainingimages, [img]), axis = 0)
        if currentPair % 100 == 0:
            print "Current pair: " + str(currentPair)

        # trainingsentences.append(sentence)
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


def createFeatDict(names, namesfile, features):
    result = {}
    for name in names:
	#print "trying to add feature to dict for: "+name
        result[name] = getImage(name, namesfile, features)
    return result

'''
Returns the RFF function of the given vector, based on the given sigma and wanted dimension
'''
def phi(wantedDimension, sigma, x):
    b = np.random.rand(wantedDimension)
    R = np.random.normal(scale = sigma*sigma, size = (len(x), wantedDimension))
    return np.dot(x,R) + b

'''
Given a matrix with each row an observation, returns the average distance to the 50th nearest neighbor
'''
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


'''
given an image sentence pair, return an array containing the concatenation of the 5 sentences in the pair
'''
def getFullSentence(imagesentencepair):
    sentences = imagesentencepair['image']['sentences']
    s = getStopwords()
    full = []
    print sentences
    for sentence in sentences:
        result = remove_common_words(sentence['tokens'], s)
        full.extend(result)
    return full


'''
Given a sentence, return a copy of that sentence, stripped of words that are in the provided stopwords
'''
def remove_common_words(sentence,stopwords):
        #s = set(stopwords.words('english'))
        stopwords.add(' ') #add spaces to stopwords
        result = []
        for word in sentence:
            if not word.lower() in stopwords and len(word)>2:
                result.append(word.lower())
        return result


'''
Given a filename, checks if the image behind that filename is bigger than 64x64
'''
def isLargeEnough(filename):
    file = filename+".jpg"
    #print file
    try:
        image = Image.open("./Flickr30kEntities/image_snippets/"+file)
    except IOError:
        
	#print "IMG NOT FOUND"
	return False
    width, height = image.size
 #   print width,height
    return (width >=400 ) and (height >= 400)


'''
Returns the image features corresponding to the provided image name
'''
def getImage(filename, file_with_names, features):
    #print filename
    file_with_names = open(file_with_names)
    line = file_with_names.readline()
    linenumber = 0
    while(not line == ""):
        if line[0:-5] == filename:
            #print 'RETURNING A FEATURE'
            return features[linenumber]
        line = file_with_names.readline()
        linenumber+=1


if __name__ == "__main__":
    names = "./Flickr30kEntities/image_snippets/images.txt"
    feats = scipy.io.loadmat("./Flickr30kEntities/image_snippets/vgg_feats.mat")['feats'].transpose()
    print "SHAPE FEAT: " + str(feats.shape)
    mainExec(names, feats)


