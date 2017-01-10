import copy

from nnet.models.DepLSTM import *
from nnet.util import *
from nnet.models.model import Model


class PredicateSenseDisamb(Model):
    def __init__(self, hps, *_):
        self.hyperparams = (hps,)
        self.test_mode = False
        self.params = list()

        # inputs ==============================================================

        sent = input_batch(2, 'int32')
        sent_mask = input_batch(2, 'float32')

        predicates = input_batch(1, 'int32')
        predicate_marks = input_batch(2, 'float32')
        predicate_pos = input_batch(1, 'int32')

        labels = input_batch(1, 'int32')
        pos_tags = input_batch(2, 'int32')

        self.inputs = [
            sent.input_var,
            sent_mask.input_var,
            predicates.input_var,
            predicate_marks.input_var,
            predicate_pos.input_var,
            labels.input_var,
            pos_tags.input_var
        ]

        word_emb_init = hps['word_embeddings'] if 'word_embeddings' in hps \
            else LI.Orthogonal()

        sent_emb = LL.EmbeddingLayer(
            sent, hps['vword'], hps['sent_edim'], name='sent_emb', W=word_emb_init)

        pred_emb = LL.EmbeddingLayer(
            predicates, hps['vword'], hps['sent_edim'], name='pred_emb')

        pos_emb = LL.EmbeddingLayer(
            pos_tags, hps['vpos'], 50, name='pos_emb'
        )

        input_emb = sent_emb
        if hps['use_predicate_as_input']:
            pred_emb_broadcast = TT.zeros_like(lo(sent_emb)) + lo(pred_emb).dimshuffle((0, 'x', 1))
            pred_emb_broadcast = LL.InputLayer((None, None, hps['sent_edim']), pred_emb_broadcast)

            input_emb = LL.ConcatLayer([pred_emb_broadcast, sent_emb], axis=2)

        if hps['use_predicate_mark']:
            region_mark = LL.dimshuffle(predicate_marks, (0, 1, 'x'))
            input_emb = LL.ConcatLayer([sent_emb, region_mark], axis=2)

        input_emb = LL.ConcatLayer([input_emb, pos_emb], axis=2)

        next_input = input_emb
        for i in range(hps['rec_layers']):
            recurrent_layer = LL.LSTMLayer
            params = {
                'incoming': next_input,
                'num_units': hps['sent_hdim'],
                'grad_clipping': 1,
                'mask_input': sent_mask,
                'peepholes': hps['peepholes'],
                'name': "sent_feature_extractor_fwd_%i" % i
            }

            fwd = recurrent_layer(**params)

            bwd_params = copy.copy(params)
            bwd_params['backwards'] = True
            bwd_params['name'] = "sent_feature_extractor_bwd_%i" % i

            bwd = recurrent_layer(**bwd_params)

            next_input = LL.ConcatLayer(incomings=[fwd, bwd], axis=2)

        context = LL.SliceLayer(next_input, indices=-1, axis=1)

        if hps['take_target_hidden']:
            targets = lo(next_input)[TT.arange(lo(next_input).shape[0]), predicate_pos.input_var]
            targets = LL.InputLayer((None, hps['sent_hdim'] * 2), targets)
            self.params += lgap(next_input)
            context = targets

        if hps['predicate_at_softmax']:
            context = LL.ConcatLayer([context, pred_emb], axis=1)
        context = LL.ConcatLayer([context, pred_emb], axis=1)

        probs = LL.DenseLayer(context, hps['vlabel'], nonlinearity=LN.softmax, name='softmax')

        cross_e = TN.categorical_crossentropy(lo(probs), lo(labels))

        self.loss = cross_e
        self.compute_loss = T.function(self.inputs, cross_e)
        self.compute_probs = T.function(self.inputs, lo(probs))

        self.test = T.function(inputs=self.inputs, outputs=cross_e)

        self.params += lgap(probs)

    def predict(self, *args):
        return self.compute_probs(*args)


if __name__ == '__main__':
    # testing stuff
    T.config.exception_verbosity = 'high'
    T.config.on_unused_input = 'ignore'
    T.config.optimizer = 'None'

    model = PredicateSenseDisamb(hps={
        'sent_edim': 2,
        'vword': 5,
        'sent_emb': 2,
        'sent_hdim': 3,
        'rec_layers': 1,
        'vlabel': 6,
        'predicate_at_softmax': True,
        'use_predicate_mark': True,
        'use_predicate_as_input': True,
        'peepholes': False,
        'take_target_hidden': True
    })

    sent_batch = np.array([[1, 2, 1], [3, 4, 1]], dtype='int32')
    predicate_marks = np.array([[0, 1, 0], [1, 0, 0]], dtype='float32')
    sent_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype='float32')
    predicates = np.array([0, 1], dtype='int32')
    labels = np.array([0, 1], dtype='int32')

    res = model.test(sent_batch, sent_mask, predicates, predicate_marks, labels)
    loss = model.compute_loss(sent_batch, sent_mask, predicates, predicate_marks, labels)
    predict = model.predict(sent_batch, sent_mask, predicates, predicate_marks, labels)
    print(res, sep='\n')
    print('\n'.join(map(str, model.params)))
