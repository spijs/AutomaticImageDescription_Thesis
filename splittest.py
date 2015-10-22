import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict


dataset = json.load(open('data/flickr30k/flickr30k/dataset.json','r'))
split = defaultdict(list)
print split
print dataset['images'][29000]['split']