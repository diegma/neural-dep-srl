from nnet.util import *

from theano.tensor.nnet import categorical_crossentropy


def create_optimizer(name, model, rate):
    params = model.get_params()
    if not isinstance(params, dict):
        raise TypeError('params must be a dict')
    params = list(params.values())

    loss = TT.sum(model.loss) / TT.sum(TT.neq(model.loss, 0))
    inputs = model.inputs

    if name == 'sgd':
        update = L.updates.sgd(loss, params, rate)
    elif name == 'adadelta':
        update = L.updates.adadelta(loss, params)
    elif name == 'rmsprop':
        update = L.updates.rmsprop(loss, params)
    elif name == 'adagrad':
        update = L.updates.adagrad(loss, params)
    elif name == 'nag':
        update = L.updates.nesterov_momentum(loss, params, rate)
    elif name == 'adam':
        update = L.updates.adam(loss, params)
    elif name == 'momentum':
        update = L.updates.momentum(loss, params, rate)
    else:
        raise NotImplementedError(name)

    return T.function(inputs=inputs, updates=update, name=name)
