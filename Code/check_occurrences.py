import json

file = json.load(open('Results/lstm30k_struct.json'))

imgblobs = file['imgblobs']

''' This script checks how many times occur right after eachother in the test set.'''
for img in imgblobs:
    candidate = img['candidate']['text']
    words = candidate.split()
    doubles = []
    for i in range(len(words)):
        if i < len(words)-2:
            doubles.append(words[i]+" "+words[i+1])
    for i in range(len(doubles)):
        if i < len(doubles) - 1 :
            try:
                rest = doubles[i+1:]
                index = rest.index(doubles[i])
                print "Double detected: ", doubles[i], "In sentence:", candidate
            except ValueError:
                pass