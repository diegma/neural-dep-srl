from nnet.util import *
from nnet.corpus import *
from nnet.converter import *
# import matplotlib.pyplot as plt
import os.path
import http.server
import threading


def run_http_server():
    http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler,
                     port=19101)


class ErrorRateTester(object):
    def __init__(self, error_computer, out, corpus):
        self.error_computer = error_computer
        self.best = float('inf')
        self.out = out
        self.errors = list()
        self.losses = list()
        self.window = 1000
        self.corpus = corpus

    def run(self, model):
        loss, errors, errors_w = self.compute_error(model)
        log('test errors: %i (weighted = %f) out of %i' %
            (errors, errors_w, self.corpus.size()))

        col = 'green' if self.best > loss else 'red'
        log('total test loss = %s' % colour(str(loss), col))

        if self.best > loss:
            self.best = loss
            pickle_model(os.path.join(self.out, 'best.bin'), model)

        log('best total test loss so far = ' + str(self.best))

        self.errors.append(errors_w)
        self.losses.append(loss)

        self.plot()

        return loss, errors, errors_w

    def compute_error(self, model):
        losses, errors, errors_w = 0., 0, 0.

        for batch in self.corpus.batches():
            loss, e, e_w = self.error_computer.compute(model, batch)
            losses += loss
            errors += e
            errors_w += e_w

        if self.error_computer.final:
            result = self.error_computer.final()
            if result is not None:
                losses = -result

        return losses, errors, errors_w

    def plot(self):
        return
        plt.figure(1)
        plt.clf()

        plt.ylabel('test weighted errors')
        plt.subplot(211)
        plt.plot(self.errors[-self.window:], 'r-')

        plt.ylabel('test mean loss')
        plt.subplot(212)
        plt.plot(self.losses[-self.window:], 'b-')

        plt.savefig('test_set_plot.png')


class NullTester(object):
    def __init__(self):
        self.best = float('inf')

    def compute_error(self, *args, **kwargs):
        return 0, 0, 0

    def run(self, *args, **kwargs):
        pass
