#!/usr/bin/env python3

import argparse
import sys
import lasagne.init

initializer = lasagne.init.Uniform()


def main():
    voc, full_embeddings = sys.argv[1], sys.argv[2]

    voc = set([line.strip() for line in open(voc, 'r')] + ['_UNK'])

    # get all embeddings from full embeddings
    embeddings = dict()
    for i, line in enumerate(open(full_embeddings, 'r')):
        parts = line.rstrip().split()
        word = parts[0]
        if word in voc:
            # print(parts[1:])
            try:
                embeddings[word] = list(map(float, parts[1:]))
            except Exception as e:
                print('cannot parse line %i' % i, file=sys.stderr)

    # estimate dim
    dim = len(list(embeddings.values())[0])
    if dim == 0:
        raise Exception('embedding dim is 0, probably parsing error')

    # init unk
    embeddings['_UNK'] = initializer((dim,))

    # handle missing embeddings
    for word in voc:
        if word not in embeddings:
            print("no embedding for %s, skipping it " % word, file=sys.stderr)
            emb = initializer((dim,))
        else:
            emb = embeddings[word]
        print(word + '\t' + ' '.join(map(str, emb)))


if __name__ == '__main__':
    main()
