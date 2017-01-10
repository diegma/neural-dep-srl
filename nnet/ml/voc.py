import argparse
import sys
from collections import Counter
from nnet.ml.formats import create_tokenizer
from multiprocessing import cpu_count, Pool


class Voc(object):
    def vocalize(self, seq):
        return [self.get_id(c) for c in list(seq)]

    def get_id(self):
        raise NotImplementedError()


class TwoWayVoc(Voc):
    def devocalize(self, seq):
        return [self.get_item(c) for c in list(seq)]

    def get_item(self):
        raise NotImplementedError()


class NullVoc(Voc):
    def __init__(self, size):
        self._size = size

    def get_id(self, entry):
        id = int(entry)
        if id > self._size - 1:
            raise Exception("bad null voc entry: %i" % id)
        return id

    def size(self):
        return self._size


class HashVoc(Voc):
    def __init__(self, size):
        self._size = size

    def get_id(self, entry):
        hash = 5381
        for c in entry:
            hash = (((hash << 5) + hash) + ord(c)) % 2147483647
        return hash % self._size

    def size(self):
        return self._size


class FileVoc(TwoWayVoc):
    def __init__(self, f):
        self._add_unks = False

        with open(f, 'r') as f:
            voc = [l[:-1] for l in f.readlines()]

            # unk, eos, bos = 'UNK', '$', '^'
            # voc += [unk, bos, eos]
            voc = ['_UNK'] + voc

            self.direct = [l.split('\t')[0] for l in voc]
            self.inverted = {id: token for token, id in enumerate(self.direct)}

    def add_unks(self):
        self._add_unks = True

    def get_item(self, id):
        return self.direct[id]

    def get_id(self, entry):
        if entry not in self.inverted:
            if self._add_unks:
                return self.unk()
            raise ValueError(
                "no such value in vocabulary and unks are disabled: %s" % entry)

        return self.inverted[entry]

    def unk(self):
        return 0

    # def bos(self):
    #     return len(self.direct) - 2
    #
    # def eos(self):
    #     return len(self.direct) - 1

    def size(self):
        return len(self.direct)

def frequency_voc(f):
    freq = dict()
    with open(f, 'r') as f:
        for line in f:
            line_split = line.split('\t')
            freq[line_split[0]] = int(line_split[1])
    return freq

def create_voc(name, *args, **kwargs):
    vocs = {
        'null': NullVoc,
        'hash': HashVoc,
        'file': FileVoc,
    }

    if name not in vocs:
        raise NotImplementedError(name)

    return vocs[name](*args, **kwargs)


class _Worker:
    def __init__(self, tokenizer):
        self.tokenize = create_tokenizer(tokenizer)

    def __call__(self, lines):
        voc = Counter()
        for line in lines:
            for t in self.tokenize(line[:-1]):
                voc[t] += 1
        return voc


def _run():
    # argparse
    parser = argparse.ArgumentParser(description='vocabulary compiler')
    parser.add_argument('-n', '--nbest', dest='nbest', default=-1, type=int)
    parser.add_argument('-f', '--print-freq', dest='print_freq',
                        action='store_true')
    parser.add_argument('--tokenizer', dest='tokenizer', required=True)
    parser.add_argument('-j', '--threads', dest='threads', default=cpu_count(),
                        type=int)
    parser.add_argument('-m', '--min-freq', dest='min_freq', type=int, default=0)
    a = parser.parse_args()

    # compile voc for each thread
    lines = sys.stdin.readlines()
    numlines = int((len(lines) / a.threads) + 1)
    pool = Pool(a.threads)
    worker = _Worker(a.tokenizer)
    result = pool.map(worker, (lines[l:l + numlines] for l in
                               range(0, len(lines), numlines)))

    # merge vocs
    voc = Counter()
    for v in result:
        voc += v

    voc = voc.most_common(a.nbest if a.nbest >= 0 else None)

    # print result
    for token, freq in voc:
        line = token
        if a.print_freq:
            line += '\t' + str(freq)
        if freq >= a.min_freq:
            print(line)


if __name__ == '__main__':
    _run()
