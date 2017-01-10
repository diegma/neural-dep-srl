#!/usr/bin/env python3

import sys
import lasagne.init

initializer = lasagne.init.Uniform()

voc = [line.strip() for line in open(sys.argv[1], 'r')] + ['_UNK']
mask = set(voc)

glove = dict()

for i, line in enumerate(open(sys.argv[2], 'r')):
    parts = line.rstrip().split()
    #print(parts[1:], file=sys.stderr)
    if parts[0] in mask:
        glove[parts[0]] = list(map(float, parts[1:]))

dim = len(list(glove.values())[0])

for word in voc:
    if word not in glove:
        print("no embeding for %s, skipping it " % word, file=sys.stderr)
        if word == '_UNK':
            emb = initializer((dim,))
            print(word + '\t' + ' '.join(map(str, emb)))
    else:
        emb = glove[word]

        print(word + '\t' + ' '.join(map(str, emb)))