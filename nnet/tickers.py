import numpy as np
import os
from copy import deepcopy
from collections import defaultdict
from nnet.util import pickle_model, log, stty_width


class Ticker(object):
    def __init__(self, trigger_count):
        self.trigger_count = trigger_count

    def tick(self, batch_count, epoch, sample_count, *args, **kwargs):
        if batch_count % self.trigger_count == 0:
            self._tick(batch_count, epoch, sample_count, *args, **kwargs)

    def _tick(self, *args, **kwargs):
        raise NotImplementedError()


class ModelDumpTicker(Ticker):
    def __init__(self, out, dump_ratio, model):
        super().__init__(dump_ratio)
        self.out = out
        self.model = model

    def _tick(self, batch_count, epoch, sample_count):
        dump_name = "epoch_%i_batch_%i" % (epoch, batch_count)
        log("dumping %s ..." % dump_name)
        pickle_model(os.path.join(self.out, dump_name), self.model)


class TesterTicker(Ticker):
    def __init__(self, model, tester, ratio):
        super().__init__(ratio)
        self.tester = tester

    def _tick(self, batch_count, epoch, sample_count):
        log('test data run:')
        self.tester.run()

