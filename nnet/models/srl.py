import copy

from nnet.util import *
from nnet.models.model import Model
from nnet.models.WordDropout import WordDropoutLayer, \
    ConditionedWordDropoutLayer

from nnet.run.srl.read_dependency import _N_LABELS
from nnet.models.GraphConvolutionalLayer import GraphConvLayer
# np.set_printoptions(threshold=np.nan)
_BIG_NUMBER = 10. ** 6.


class BioSRL(Model):
    def __init__(self, hps, *_):
        self.hyperparams = (hps,)
        self.test_mode = False

        # inputs ==============================================================

        sent = input_batch(2, 'int32')
        p_sent = input_batch(2, 'int32')
        freq = input_batch(2, 'float32')
        pos_tags = input_batch(2, 'int32')

        sent_mask = input_batch(2, 'float32')
        target_idx = input_batch(1, 'int32')
        targets_idx = input_batch(2, 'int32')

        sent_pred_lemmas_idx = input_batch(2, 'int32')

        frames = input_batch(2, 'int32')
        region_mark = input_batch(2, 'float32')
        labels_voc = input_batch(2, 'int32')
        labels_voc_mask = input_batch(2, 'float32')

        adj_arcs_in = input_batch(2, 'int32')
        adj_arcs_out = input_batch(2, 'int32')
        adj_lab_in = input_batch(2, 'int32')
        adj_lab_out = input_batch(2, 'int32')
        mask_in = input_batch(2, 'float32')
        mask_out = input_batch(2, 'float32')
        mask_loop = input_batch(2, 'float32')

        labels = input_batch(2, 'int32')

        self.inputs = [
            sent.input_var,
            p_sent.input_var,
            pos_tags.input_var,
            sent_mask.input_var,
            target_idx.input_var,
            frames.input_var,
            labels_voc.input_var,
            labels_voc_mask.input_var,
            freq.input_var,
            region_mark.input_var,
            sent_pred_lemmas_idx.input_var,
            adj_arcs_in.input_var, adj_arcs_out.input_var,
            adj_lab_in.input_var, adj_lab_out.input_var,
            mask_in.input_var, mask_out.input_var,
            mask_loop.input_var,
            labels.input_var
        ]

        bsize = lo(sent).shape[0]
        bidir_dim = hps['sent_hdim'] * 2
        src_hid_dim = 2 * bidir_dim
        # determine input embeddings size
        emb_len = 2 * hps['sent_edim']
        if hps['pos']:
            emb_len += hps['pos_edim']

        if hps['p_lemmas']:
            emb_len += hps['sent_edim']

        emb_len += 1  # this is for the region mark that can be always zero



        pos_d = pos_tags

        drop_mask = WordDropoutLayer(sent, freq, hps['alpha'])
        sent_d = LL.ElemwiseMergeLayer([sent, drop_mask], TT.mul)
        fixed_mask = ConditionedWordDropoutLayer(p_sent, drop_mask)
        fixed_sent_d = LL.ElemwiseMergeLayer([p_sent, fixed_mask], TT.mul)

        # compute embeddings
        # emb.shape == (bsize, time, sent_edim)

        word_emb_init = hps['word_embeddings'] if 'word_embeddings' in hps \
            else LI.Orthogonal()

        sent_emb = LL.EmbeddingLayer(
            sent_d, hps['vword'], hps['sent_edim'], name='sent_emb')


        if hps['pos']:
            pos_emb = LL.EmbeddingLayer(
                pos_d, hps['vpos'], hps['pos_edim'], name='pos_emb',
                W=LI.Orthogonal())
            sent_emb = LL.ConcatLayer(incomings=[pos_emb, sent_emb], axis=2)


        if hps['p_lemmas']:
            p_lemmas_emb = LL.EmbeddingLayer(
                sent_pred_lemmas_idx, hps['vframe'], hps['sent_edim'], name='predicate_emb',
                W=LI.Orthogonal())

            sent_emb = LL.ConcatLayer(incomings=[p_lemmas_emb, sent_emb], axis=2)

        # take pretrained embeddings and fix them
        sent_emb_fixed = LL.EmbeddingLayer(
            fixed_sent_d, word_emb_init.shape[0], hps['sent_edim'],
            name='sent_emb_fixed',
            W=word_emb_init)
        sent_emb_fixed.params[sent_emb_fixed.W].remove('trainable')


        region_mark = LL.dimshuffle(region_mark, (0, 1, 'x'))
        sent_emb = LL.ConcatLayer(incomings=[sent_emb, region_mark],
                                  axis=2)

        # concat them with trainable embeddings
        sent_emb_concat = LL.ConcatLayer(incomings=[sent_emb, sent_emb_fixed],
                                         axis=2)
        sent_emb = sent_emb_concat


        next_input = sent_emb
        for i in range(hps['rec_layers']):
            recurrent_layer = LL.LSTMLayer
            params = {
                'incoming': next_input,
                'num_units': hps['sent_hdim'],
                'grad_clipping': 1,
                'mask_input': sent_mask,
                'peepholes': False,
                'name': "sent_feature_extractor_fwd_%i" % i
            }

            fwd = recurrent_layer(**params)

            bwd_params = copy.copy(params)
            bwd_params['backwards'] = True
            bwd_params['name'] = "sent_feature_extractor_bwd_%i" % i

            bwd = recurrent_layer(**bwd_params)
            next_input = LL.ConcatLayer(incomings=[fwd, bwd], axis=2)

        for i in range(hps['gc_layers']):
            g_params = {
                'incoming': next_input,
                'num_units': hps['gcb_hdim'],
                'arc_tensor_in': lo(adj_arcs_in),
                'arc_tensor_out': lo(adj_arcs_out),
                'label_tensor_in': lo(adj_lab_in),
                'label_tensor_out': lo(adj_lab_out),
                'mask_in': lo(mask_in),
                'mask_out': lo(mask_out),
                'mask_loop': lo(mask_loop),
                'num_labels': _N_LABELS,
                'batch_size': bsize,
                'in_arcs': hps['in_arcs'],
                'out_arcs': hps['out_arcs'],
                'dropout': hps['gc_dropout'],
                'mask_input': sent_mask,
                'grad_clipping': 1,
                'name': "graph_conv_%i" % i
            }
            next_input = GraphConvLayer(**g_params)

        # bidir.shape == (bsize, time, bidir_dim)
        # bidir = LL.ConcatLayer(incomings=[fwd, bwd], axis=2)
        bidir = next_input
        bidir_hid = lo(bidir)
        # bidir_hid = theano.printing.Print("bidir_hid=")(bidir_hid)
        # combine hidden states with targets ==================================

        # targets.shape = (bsize, bidir_dim)
        targets = bidir_hid[
            TT.arange(bidir_hid.shape[0]), target_idx.input_var]
        # targets = theano.printing.Print("targets=")(targets)
        targets += TT.zeros_like(bidir_hid.dimshuffle((1, 0, 2)))
        # targets = theano.printing.Print("targets=")(targets)
        targets = targets.dimshuffle((1, 0, 2))
        # targets = theano.printing.Print("targets=")(targets)


        bidir_hid_ = TT.concatenate([bidir_hid, targets], axis=2)


        # obtain the roles representation
        frame_emb = LL.EmbeddingLayer(
            frames, hps['vframe'], hps['frame_edim'], name='frame_emb',
            W=LI.Orthogonal())
        role_emb = LL.EmbeddingLayer(
            labels_voc, hps['vbio'], hps['role_edim'], name='role_emb',
            W=LI.Orthogonal())

        # combine role and frame ==============================================

        role_concat = LL.ConcatLayer(incomings=[frame_emb, role_emb], axis=2)

        # flatten things out, since dense layer cannot work with 3d tensors
        role_hid = LL.ReshapeLayer(
            role_concat, (
                lo(labels_voc).shape[0] * lo(labels_voc).shape[1],
                hps['frame_edim'] + hps['role_edim']))

        # add nonlinearity
        role_hid = LL.DenseLayer(
            incoming=role_hid, nonlinearity=LN.rectify, num_units=src_hid_dim,
            name="role_hid")
        # shape back to (bsize, roles, src_hid_dim)
        role_hid = LL.ReshapeLayer(
            role_hid,
            (lo(labels_voc).shape[0], lo(labels_voc).shape[1], src_hid_dim))

        # (bsize, time, roles)

        # bidir_hid.shape = (B, time, H)
        # role_hid = (B, roles, H)
        # dots.shape = (B, time, roles)
        o_dots = TT.batched_tensordot(bidir_hid_, lo(role_hid), axes=(2, 2))

        # compute dot products between words and roles ========================

        # mask out roles that are not relevant to corresponding sentences
        sub = (labels_voc_mask.input_var - TT.ones_like(
            labels_voc_mask.input_var)) * _BIG_NUMBER
        # dimshuffle dots so that sub can be properly broadcasted
        dots = o_dots.dimshuffle((1, 0, 2))
        # apply mask
        dots += sub
        # shuffle dots to original shape
        dots = dots.dimshuffle((1, 0, 2))
        potentials = TT.reshape(dots, (bsize * dots.shape[1], dots.shape[2]))
        probs = TT.nnet.softmax(potentials)

        # Deterministic
        # This block below is necessary to turn off word drop out at test time
        bidir_hid_de = LL.get_output(bidir, deterministic=True)
        targets_de = bidir_hid_de[
            TT.arange(bidir_hid_de.shape[0]), target_idx.input_var]
        targets_de += TT.zeros_like(bidir_hid_de.dimshuffle((1, 0, 2)))
        targets_de = targets_de.dimshuffle((1, 0, 2))
        bidir_hid_de = TT.concatenate([bidir_hid_de, targets_de],
                                              axis=2)

        role_hid_de = LL.ReshapeLayer(role_hid,
                                      (lo(labels_voc).shape[0],
                                       lo(labels_voc).shape[1], src_hid_dim))
        role_hid_de = LL.get_output(role_hid_de, deterministic=True)
        o_dots_de = TT.batched_tensordot(bidir_hid_de, role_hid_de,
                                         axes=(2, 2))
        sub_de = (labels_voc_mask.input_var - TT.ones_like(
            labels_voc_mask.input_var)) * _BIG_NUMBER
        dots_de = o_dots_de.dimshuffle((1, 0, 2))
        dots_de += sub_de
        dots_de = dots_de.dimshuffle((1, 0, 2))
        potentials_de = TT.reshape(dots_de, (
            bsize * dots_de.shape[1], dots_de.shape[2]))
        probs_det = TT.nnet.softmax(potentials_de)

        # compute loss ========================================================
        flat_labels = TT.flatten(labels.input_var, 1)

        cross_e = TT.nnet.categorical_crossentropy(probs, flat_labels)
        self.loss = mask_loss(cross_e, sent_mask)

        # functions
        self.compute_probs = T.function(self.inputs[:-1], probs_det)
        self.compute_loss = T.function(self.inputs, self.loss)
        self.test = T.function(self.inputs, cross_e)
        b_params = LL.get_all_params(bidir, trainable=True)
        r_params = LL.get_all_params(role_hid, trainable=True)
        self.params = r_params + b_params

    def predict(self, *args):
        return self.compute_probs(*args)

