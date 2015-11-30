if __name__ == "__main__":
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
    words = dict.keys()
    f = open("dictionary.txt", 'w+')
    for w in words:
        f.writelines(w+'\n')