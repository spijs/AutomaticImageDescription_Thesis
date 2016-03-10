__author__ = 'spijs'

__author__ = 'Wout & thijs'


from imagernn.data_provider import getDataProvider


def create_vocabulary(params):
    dataset = 'flickr30k'
    dataprovider = getDataProvider(dataset)
    img_sentence_pair_generator = dataprovider.iterImageSentencePair()
    dict = {}
    result = []
    for pair in img_sentence_pair_generator:
        sentence = pair['sentence']['tokens']
        for word in sentence:
            word = word.decode('utf-8').lower()
            if(not word in dict):
                dict[word]=1
            else:
                dict[word]+=1
    for word in dict:
            if(dict[word] >= 5):
                result.append(word)
    f = open("vocabulary.txt", "w+")
    for w in result:
        f.writelines(w+'\n')
    print('created vocabulay')
