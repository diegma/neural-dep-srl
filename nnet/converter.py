from nnet.util import *
import numpy as np


class LmConverter(object):
    def __init__(self, voc):
        self.voc = voc

    def convert(self, batch):
        batch = [self.voc.vocalize(s) + [self.voc.eos()] for s in batch]
        return mask_batch(batch)


class Seq2SeqConverter(object):
    def __init__(self, voc_in, voc_out):
        self.voc_in = voc_in
        self.voc_out = voc_out

    @classmethod
    def from_single_voc(cls, voc):
        return cls(voc, voc)

    def convert(self, batch):
        src, dst = list(zip(*batch))
        src = [list(reversed(self.voc_in.vocalize(s))) + [self.voc_in.eos()]
               for s in src]
        # add EOS sentinel
        dst = [self.voc_out.vocalize(s) + [self.voc_out.eos()] for s in dst]

        src_batch, src_mask = mask_batch(src)
        dst_batch, dst_mask = mask_batch(dst)

        return src_batch, dst_batch, src_mask, dst_mask


class ClassifierConverter(object):
    def __init__(self, voc):
        self.voc = voc

    def convert(self, batch):
        def scale(w):
            return w if w > 0.9 else 0.2

        features, labels, weights = list(zip(*batch))
        features = [self.voc.vocalize(f) for f in features]
        weights = [np.float32(w) for w in weights]
        return [features, labels, weights]
