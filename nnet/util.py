from theano import config, shared
from theano.gradient import grad_clip
import functools
import math
import numpy as np
import os
import pickle
import struct
import sys
import time
from collections import OrderedDict
from theano import scan, scan_module
from theano.tensor import dot
import logging
import theano as T
import theano.tensor as TT
import theano.tensor.nnet as TN
import lasagne as L
import lasagne.layers as LL
from lasagne.utils import create_param
from lasagne.layers import get_output as lo
from lasagne.objectives import categorical_crossentropy
import lasagne.nonlinearities as LN
import lasagne.random
import lasagne.updates as LU
import lasagne.init as LI
import lasagne.objectives as LO
from lasagne.layers import get_output as lo


# from misspell.edit_dist import *


# Numerics -------------------------------------------------------------------


def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - TT.log(TT.sum(TT.exp(xdev), axis=1, keepdims=True))


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -TT.sum(targets * log_predictions, axis=1)


# Serialization --------------------------------------------------------------


def escape(path):
    return path.replace(' ', '_')


def save_model(folder, model, txt=False):
    if not os.path.exists(folder):
        os.makedirs(folder)

    temps = {}
    for k, v in model.get_params().items():
        filename = os.path.join(folder, escape(k) + '.npy')
        temps[k] = filename
        np.savetxt(filename, v.get_value())

    # rm text files, dump the contents to a single file
    if not txt:
        with open(os.path.join(folder, 'model.bin'), 'wb') as out:
            out.write(struct.pack('i', len(temps)))
            for k, filename in temps.items():
                out.write(struct.pack('i', len(k)))
                out.write(str.encode(k))
                with open(filename, 'r') as in_:
                    lines = in_.readlines()
                    nCols = len(lines[0].strip().split()) if lines else 0
                    out.write(struct.pack('i', 4 * (2 + len(lines) * nCols)))
                    out.write(struct.pack('i', len(lines)))
                    out.write(struct.pack('i', nCols))
                    for line in lines:
                        for cell in line.strip().split():
                            out.write(struct.pack('f', float(cell)))
                os.remove(filename)


def pickle_model(file, model):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.makedirs(dir)

    pickle.dump(model, open(file, 'wb'))


def unpickle_model(file):
    return pickle.load(open(file, 'rb'))


def load_model(folder, model):
    for k, v in model.get_params().items():
        param = np.load(os.path.join(folder, escape(k) + '.npy'))
        v.set_value(param)


# Misc -----------------------------------------------------------------------

def pad(sample, n):
    res = sample[:n]
    padding = [-1] * (n - len(res))
    return res + padding


def stty_width():
    # rows, cols = os.popen('stty size', 'r').records().split()
    # return int(cols)
    return 50


# Lasagne --------------------------------------------------------------------

def make_tensor(dim, dtype):
    return TT.TensorType(dtype=dtype, broadcastable=(False,) * dim)()


def input_batch(dim, dtype):
    ttype = TT.TensorType(dtype, ((False,) * dim))
    return LL.InputLayer(shape=((None,) * dim), input_var=ttype(None))


def li(x, odim):
    shape = ((None,) * (x.ndim - 1)) + (odim,)
    return LL.InputLayer(shape=shape, input_var=x)


def mask_loss(loss, mask):
    return loss * lo(LL.FlattenLayer(mask, 1))


def widen(x):
    return x.dimshuffle(x.shape + ('x',))


def broadcast_vec(x, n):
    form = TT.ones((n, 1))
    return TT.dot(form, x)


def lgof(l, x):
    return l.get_output_for(x)


def lgap(l):
    return LL.get_all_params(l)


def ls(l, x):
    return lo(l).shape[x]


def concat_tensors(xs, axis=0):
    broad = [x.dimshuffle('x', 0, 1) for x in xs]
    return TT.concatenate(broad, axis=axis)


def mask_batch(batch):
    max_len = len(max(batch, key=len))
    mask = np.zeros((len(batch), max_len))
    padded = np.zeros((len(batch), max_len))
    for i in range(len(batch)):
        mask[i, :len(batch[i])] = 1
        for j in range(len(batch[i])):
            padded[i, j] = batch[i][j]

    return padded.astype('int32'), mask.astype(T.config.floatX)


def get_number_of_params(model):
    return sum(
        [functools.reduce(lambda acc, y: acc * y, p.get_value().shape, 1)
         for p in model.get_params().values()]
    )


# ----------------------------------------------------------------------------


def nnjm_samples(q, c, win):
    q += '$'
    c += '$'

    q, c = align(q, c)

    def pad(s, n, direct=True):
        s = s[-n:] if direct else s[:n]
        res = '^' * (n - len(s)) + s
        if len(res) != n:
            raise Exception(s, direct)
        return res

    for i in range(len(c)):
        if c[i] == '#':
            clc = c[:i].replace('#', '')
            crc = c[i:].replace('#', '')
            char = clc[-1] if clc else crc[0]
            clc = clc[:-1]
        else:
            clc = c[:i].replace('#', '')
            char = c[i]

        qlc = q[:i].replace('#', '')
        qrc = q[i:].replace('#', '')

        sample = pad(qlc, win) + pad(qrc, win, False) + pad(clc, win)

        yield sample, char, q[i]


# def align(q, c):
#     def sub(c):
#         return c if c else '#'
#
#     seq = edit_sequence(q, c)
#
#     src, dst, w, op = zip(*seq)
#     src = ''.join([sub(c) for c in src])
#     dst = ''.join([sub(c) for c in dst])
#
#     return src, dst


def nnjm_prefix(q, c, win):
    q += '$'
    c += '$'

    def pad(s, n):
        return '^' * (win - len(s)) + s

    for i in range(len(c)):
        lc = pad(c[max(i - win, 0):i], win)
        j = int((float(i) / len(c)) * len(c))
        lq = pad(q[max(j - win, 0):j], win)
        rq = pad(q[j:min(j + win, len(q))], win)

        sample = lq + rq + lc

        yield sample, c[i], c[i]


def log(*args, **kwargs):
    print(file=sys.stderr, flush=True, *args, **kwargs)


def make_colour_f():
    _COLOURS = {
        'blue': '94m',
        'green': '92m',
        'red': '91m',
        'yellow': '93m',
        'white': '37m',
    }

    _FORMATS = {
        'header': '95m',
        'bold': '1m',
        'underline': '4m'
    }

    def colour(s, colour, formatting=None):
        if formatting and formatting not in _FORMATS:
            raise ValueError(formatting)

        if colour not in _COLOURS:
            raise ValueError(colour)
        res = '\033[' + _COLOURS[colour] + s + '\033[0m'
        if formatting:
            return '\033[' + _FORMATS[formatting] + res
        return res

    return colour


colour = make_colour_f()


def dash_line():
    log("-" * stty_width())


def update_model(old_model, new_model):
    old_model.__setstate__(new_model.__getstate__())


class TickTack:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tick = time.time()
        log(self.name)

    def __exit__(self, type, value, traceback):
        tack = time.time()
        diff = (tack - self.tick) / 60.0
        log('finished, took %f sec' % diff)


def parse_word_embeddings(embeddings):
    res = []

    for line in open(embeddings, 'r'):
        emb = map(float, line.strip().split()[1:])
        res.append(list(emb))

    return np.array(res, dtype='float32')
