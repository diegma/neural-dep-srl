from nnet.util import *


class Model(object):
    def __init__(self, *args, **kwargs):
        self.test_mode = False

    def __getstate__(self):
        return [p.get_value() for p in self.params]

    def __setstate__(self, state):
        for i, param in enumerate(self.params):
            param.set_value(state[i])

    def __getnewargs__(self):
        return self.hyperparams

    def __new__(cls, *args, **kwargs):
        logging.debug('created %s with params %s' % (str(cls), str(args)))

        instance = super(Model, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance

    def get_params(self):
        return {param.name: param for param in self.params}

    def test_mode_on(self):
        self.test_mode = True

    def test_mode_off(self):
        self.test_mode = False
