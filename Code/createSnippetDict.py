import os
from nltk.stem.porter import *
from PIL import Image

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
    #print width,height
    return (width >= 150) and (height >= 150)

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

if __name__ == "__main__":
    dict = {}
    result = {}
    stopwords = getStopwords()
    current = 0
    for dirname, dirnames, filenames in os.walk('./Flickr30kEntities/sentence_snippets'):
        for filename in filenames:
            current += 1
        if current % 1000 == 0:
            print "Preprocessing sentence: " + str(current)
        f= open('./Flickr30kEntities/sentence_snippets/'+filename)
        line = f.readline()
        sentenceid = 1
        # print filename
        while not (line == ""):
            if isLargeEnough(filename[0:-4]+'_'+str(sentenceid)):
                for word in line.split():
                    word = stem(word.decode('utf-8')).lower()
                    if (not word in stopwords):
                        if(not word in dict):
                            dict[word]=1
                        else:
                            dict[word]+=1
            line = f.readline()
        sentenceid += 1
        for word in dict:
            if(dict[word] >= 5):
                result[word]=dict[word]
    words = result.keys()
    f = open("dictionary.txt", 'w+')
    for w in words:
        f.writelines(w+'\n')
