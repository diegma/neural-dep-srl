import argparse
import nnet.models
import numpy as np
import sys
import lasagne as L
import theano as T
from nnet.util import *
from nnet.optimization import *
from nnet.corpus import *
import signal
from nnet.training import *
from nnet.testing import *
from nnet.tickers import *


class Runner(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description="neural networks trainer")

        parser.add_argument("--debug", help="print debug info",
                            action="store_true", default=False)
        parser.add_argument("--test", help="validation set")
        parser.add_argument("--train", help="training set", required=False)
        parser.add_argument("--batch", help="batch size", default=128,
                            type=int)
        parser.add_argument("--epochs", help="n of epochs",
                            default=sys.maxsize,
                            type=int)
        parser.add_argument("--seed", help="RNG seed", default=42, type=int)
        parser.add_argument("--optimizer", default="adadelta")
        parser.add_argument("--out", help="output dir", default="out")
        parser.add_argument("--dump-ratio", default=200000, type=int)
        parser.add_argument("--log-level", default="ERROR")
        parser.add_argument("--finetune", help="pretrained model path")
        parser.add_argument("--early-stop", action="store_true")
        parser.add_argument("--rate", default=0.01, type=float)
        parser.add_argument("--tester-ratio", default=5000, type=int)
        parser.add_argument("--dump-graph", action="store_true")
        parser.add_argument("--backoff-threshold", default=0, type=int)
        parser.add_argument("--dbg-print-rate", help="in BATCHES", type=int,
                            default=5000)
        parser.add_argument("--soft-warnings", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--test-only", action="store_true", required=False)
        parser.add_argument("--test-model", required=False)

        self.add_special_args(parser)
        self.a = parser.parse_args()

        if not (self.a.train or self.a.test_only):
            parser.error('either specify --train or --test-only')

        if self.a.test_only and not self.a.test:
            parser.error('specify --test')

    def load_model(self):
        raise NotImplemented()

    def get_reader(self):
        raise NotImplemented()

    def get_tester(self):
        raise NotImplemented()

    def get_parser(self):
        raise NotImplemented()

    def get_converter(self):
        raise NotImplemented()

    def get_tester(self):
        raise NotImplemented()

    def add_special_args(self, parser):
        raise NotImplemented()

    def run(self):
        a = self.a

        np.random.seed(a.seed)
        L.random.set_rng(np.random)

        if a.soft_warnings:
            T.config.on_unused_input = 'warn'
            T.config.disconnected_inputs = 'warn'

        tester = NullTester() if not a.test else self.get_tester()
        if a.test_only:
            tester.run(unpickle_model(a.test_model))
            return

        if a.profile:
            T.config.profile = True
            T.config.profile_memory = True

        if a.debug:
            T.config.allow_gc = False
            T.config.mode = 'FAST_RUN'
            T.config.linker = 'c'
            T.optimizer = 'none'
            # T.config.compute_test_value = 'warn'
            # T.config.exception_verbosity = 'high'
        
        if a.finetune:
            log('init model from ' + a.finetune)
            model = unpickle_model(a.finetune)
        else:
            model = self.load_model()

        log('model loaded, there are %i sets of params' % len(model.get_params()))

        # create optimizers
        with TickTack('compiling optimizers...') as _:
            o_name = a.optimizer
            optimizers = [(('%s_%f' % (o_name, a.rate)).rstrip('0'),
                           create_optimizer(name=o_name, model=model,
                                            rate=a.rate))]
            if a.test and a.backoff_threshold:
                for i in range(1, 10):
                    rate = a.rate / (2 ** i)
                    op = create_optimizer(name='sgd', model=model, rate=rate)
                    name = ('sgd_%f' % rate).rstrip('0')
                    optimizers.append((name, op))

        log('optimizers in use: %s' % ', '.join(list(zip(*optimizers))[0]))

        # create training set
        log('loading corpus from %s' % a.train)

        train_set = Corpus(
            parser=self.get_parser(),
            batch_size=a.batch,
            path=a.train,
            reader=self.get_reader()
        )

        tickers = [ModelDumpTicker(a.out, a.dump_ratio, model)]

        def sig_handler(sig, frame):
            last_model = 'last.bin'
            print('SIG %s detected, dumping the model as %s...'
                  % (str(sig), last_model))
            pickle_model(os.path.join(a.out, 'last.bin'), model)
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        log('init complete\n')

        train(
            model=model,
            train_set=train_set,
            optimizers=optimizers,
            epochs=a.epochs,
            converter=self.get_converter(),
            tickers=tickers,
            tester=tester,
            threshold=a.backoff_threshold,
            dbg_print_rate=a.dbg_print_rate
        )

        log('dumping final model...')
        pickle_model(os.path.join(a.out, 'final.bin'), model)
