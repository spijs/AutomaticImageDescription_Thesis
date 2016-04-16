__author__ = 'spijs'

import pickle
import operator

def main():
    input = open('idf.p','rb')
    idf = pickle.load(input)
    sorted_idf = sorted(idf.items(), key=operator.itemgetter(1))
    output = open('sorted_idf.txt','w')
    for pair in sorted_idf:
        print "Word: " + pair[0] + " IDF: " + str(pair[1])
        output.write(pair[0] + ": " + str(pair[1])+"\n")
    output.close()
if __name__ == "__main__":
    main()