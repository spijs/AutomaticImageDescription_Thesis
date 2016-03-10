__author__ = 'spijs'

__author__ = 'Wout & thijs'


from imagernn.data_provider import getDataProvider
import json

def create_perturbed_json():
    dataset = json.load(open('data/flickr30k/dataset.json', 'r'))
    new_images = []
    # group images by their train/val/test split into a dictionary -> list structure
    for img in dataset['images']:
        split = img['split']
        tokens = img['tokens']
        if split=='train':
            img['tokens'] = _perturbe_tokens(tokens)
        new_images.append(img)
    dict = {'images':new_images}
    with open('data/flickr30k/pert_dataset.json', 'w') as fp:
        json.dump(dict, fp)



def _perturbe_tokens(tokens):
    return tokens







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
