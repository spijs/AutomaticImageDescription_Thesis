__author__ = 'spijs'

__author__ = 'Wout & thijs'


from imagernn.data_provider import getDataProvider
import json
import random

def create_perturbed_json(thresh):
    dataset = json.load(open('data/flickr30k/dataset.json', 'r'))
    new_images = []
    # group images by their train/val/test split into a dictionary -> list structure
    for img in dataset['images']:
        split = img['split']
        sentences = img['sentences']
        for sentence in sentences:
            tokens = sentence['tokens']
            if split=='train':
                sentence['tokens'] = _perturb_tokens(tokens,thresh)
        img['sentences'] = sentences
        new_images.append(img)
    dict = {'images':new_images}
    with open('data/flickr30k/pert_dataset.json', 'w') as fp:
        json.dump(dict, fp)



def _perturb_tokens(tokens, thresh):
    new_tokens =[]
    for token in tokens:
        r = random.randint(0,99)
        if r < thresh:
            new_tokens.append(_pick_random_word())
        else:
            new_tokens.append(token)
    return new_tokens

def _pick_random_word():
    r = random.randint(0,len(vocabulary)-1)
    return vocabulary[r]


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
            if(dict[word] >= 1):
                result.append(word)
    f = open("vocabulary.txt", "w+")
    for w in result:
        f.writelines(w+'\n')
    print('created vocabulary')

def read_vocabulary(filename='vocabulary.txt'):
    result = []
    voc = open(filename)
    line = voc.readline()
    while line:
        result.append(line[0:-1])
        line = voc.readline()
    return result



vocabulary = read_vocabulary()
create_perturbed_json(15)
