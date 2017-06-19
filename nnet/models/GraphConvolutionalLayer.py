import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
import lasagne.layers as LL
import lasagne.updates as LU

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers import helper
from lasagne.layers.recurrent import Gate
import lasagne.objectives as LO
from lasagne.layers import get_output as lo
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

_BIG_NUMBER = 10. ** 6.



class GraphConvLayer(MergeLayer):



    def __init__(self, incoming, num_units,
                 arc_tensor_in, arc_tensor_out,
                 label_tensor_in, label_tensor_out,
                 mask_in, mask_out,  # batch* t, degree
                 mask_loop,
                 num_labels,
                 in_arcs,
                 out_arcs,
                 dropout,
                 batch_size=100,
                 gradient_steps=-1,
                 grad_clipping=0,
                 mask_input=None,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # initial cell state. We will just provide the layer input as incomings,
        # unless a mask input, initial hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(GraphConvLayer, self).__init__(incomings, **kwargs)


        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.arc_tensor_in = arc_tensor_in  # shape [batch, time, degree] or [batch *time * degree]
        self.arc_tensor_out = arc_tensor_out
        self.label_tensor_in = label_tensor_in  # shape [batch, time, degree]
        self.label_tensor_out = label_tensor_out
        self.batch_size = batch_size
        self.num_units = num_units
        # self.m_degree = m_degree
        self.mask_in = mask_in
        self.mask_out = mask_out
        self.mask_loop = mask_loop
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        self.retain = 1. - dropout
        self.num_inputs = input_shape[2]
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.num_labels = num_labels

        if in_arcs:
            self.W_in = self.add_param(
                init.GlorotNormal(), (self.num_inputs, self.num_units), name="W_in",
                trainable=True, regularizable=True)

            self.V_in = self.add_param(
                init.Constant(0.), (num_labels, self.num_units), name="V_in",
                trainable=True, regularizable=True)

        if out_arcs:
            
            self.W_out = self.add_param(
                init.GlorotNormal(), (self.num_inputs, self.num_units), name="W_out",
                trainable=True, regularizable=True)

            self.V_out = self.add_param(
                init.Constant(0.), (num_labels, self.num_units), name="V_out",
                trainable=True, regularizable=True)

        self.W_self_loop = self.add_param(
                init.GlorotNormal(), (self.num_inputs, self.num_units), name="W_self_l",
                trainable=True, regularizable=True)

        if in_arcs:
            self.W_in_gate = self.add_param(
                init.Uniform(), (self.num_inputs, 1), name="W_in_gate",
                trainable=True, regularizable=True)
            self.V_in_gate = self.add_param(
                init.Constant(1.), (num_labels, 1), name="V_in_gate",
                trainable=True, regularizable=True)

        if out_arcs:
            self.W_out_gate = self.add_param(
                init.Uniform(), (self.num_inputs, 1), name="W_out_gate",
                trainable=True, regularizable=True)

            self.V_out_gate = self.add_param(
                init.Constant(1.), (num_labels, 1), name="V_out_gate",
                trainable=True, regularizable=True)

        self.W_self_loop_gate = self.add_param(
            init.Uniform(), (self.num_inputs, 1), name="W_self_l_gate",
            trainable=True, regularizable=True)



    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]

        return input_shape[0], input_shape[1], self.num_units


    # this method uses softmax as a gate for each arc
    def get_output_for(self, inputs, deterministic=False, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None

        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        num_batch, seq_len, _ = input.shape
        max_degree = 1
        input_ = input.reshape(
            (num_batch * seq_len, self.num_inputs))  # [b* t, h]
        if self.in_arcs:

            input_in = T.dot(input_, self.W_in)  # [b* t, h] * [h,h] = [b*t, h]
            first_in = input_in[self.arc_tensor_in[0] * seq_len + self.arc_tensor_in[1]]  # [b* t* 1, h]

            second_in = self.V_in[self.label_tensor_in[0]]  # [b* t* 1, h]
            in_ = (first_in + second_in).reshape((num_batch, seq_len, 1, self.num_units))

            # compute gate weights
            input_in_gate = T.dot(input_, self.W_in_gate)  # [b* t, h] * [h,h] = [b*t, h]
            first_in_gate = input_in_gate[self.arc_tensor_in[0] * seq_len + self.arc_tensor_in[1]]  # [b* t* 1, h]
            second_in_gate = self.V_in_gate[self.label_tensor_in[0]]
            in_gate = (first_in_gate + second_in_gate).reshape((num_batch, seq_len, 1))

            max_degree += 1
        if self.out_arcs:
            
            input_out = T.dot(input_, self.W_out)  # [b* t, h] * [h,h] = [b* t, h]

            first_out = input_out[self.arc_tensor_out[0] * seq_len + self.arc_tensor_out[1]]  # [b* t* mxdeg, h]
            second_out = self.V_out[self.label_tensor_out[0]]
            degr = T.cast(first_out.shape[0] / num_batch / seq_len, dtype='int32')
            max_degree += degr

            out_ = (first_out + second_out).reshape((num_batch, seq_len, degr, self.num_units))

            # compute gate weights
            input_out_gate = T.dot(input_, self.W_out_gate)  # [b* t, h] * [h,h] = [b* t, h]
            first_out_gate = input_out_gate[self.arc_tensor_out[0] * seq_len + self.arc_tensor_out[1]]  # [b* t* mxdeg, h]
            second_out_gate = self.V_out_gate[self.label_tensor_out[0]]
            out_gate = (first_out_gate + second_out_gate).reshape((num_batch, seq_len, degr))

            


        # same_input = input.dimshuffle(0, 1, 'x', 2)
        same_input = T.tensordot(input, self.W_self_loop, axes=[2, 0]).dimshuffle(0, 1, 'x', 2)

        same_input_gate = T.tensordot(input, self.W_self_loop_gate, axes=[2, 0]).reshape((num_batch, seq_len)).dimshuffle(0, 1, 'x')


        if self.in_arcs and self.out_arcs:
            potentials = T.concatenate([in_, out_, same_input], axis=2)  # [b, t,  mxdeg, h]
            
            potentials_gate = T.concatenate([in_gate, out_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]


            mask_soft = T.concatenate([self.mask_in, self.mask_out, self.mask_loop], axis=1)  # [b* t, mxdeg]

        elif self.out_arcs:
            potentials = T.concatenate([out_, same_input], axis=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = T.concatenate([out_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]
            mask_soft = T.concatenate([self.mask_out, self.mask_loop], axis=1)  # [b* t, mxdeg]

        elif self.in_arcs:
            potentials = T.concatenate([in_, same_input], axis=2)  # [b, t,  2*mxdeg+1, h]
            potentials_gate = T.concatenate([in_gate, same_input_gate], axis=2)  # [b, t,  mxdeg, h]
            mask_soft = T.concatenate([self.mask_in, self.mask_loop], axis=1)  # [b* t, mxdeg]

        potentials_ = potentials.dimshuffle(3, 0, 1, 2)  # [h, b, t, mxdeg]

        potentials_resh = potentials_.reshape((self.num_units,
                                               self.batch_size * seq_len,
                                               max_degree))  # [h, b * t, mxdeg]


        potentials_r = potentials_gate.reshape((self.batch_size * seq_len,
                                                  max_degree))  # [h, b * t, mxdeg]
        # calculate the gate
        probs_det_ = T.nnet.sigmoid(potentials_r) * mask_soft  # [b * t, mxdeg]
        potentials_masked = potentials_resh * mask_soft * probs_det_  # [h, b * t, mxdeg]

        
        if self.retain == 1 or deterministic:
            pass
        else:
            drop_mask = self._srng.binomial(potentials_resh.shape[1:], p=self.retain, dtype=input.dtype)
            potentials_masked /= self.retain
            potentials_masked *= drop_mask



        potentials_masked_ = potentials_masked.sum(axis=2)  # [h, b * t]

        potentials_masked__ = T.switch(potentials_masked_ > 0, potentials_masked_, 0) # ReLU

        result_ = potentials_masked__.dimshuffle(1, 0)   # [b * t, h]
        result_ = result_.reshape((self.batch_size, seq_len, self.num_units))  # [ b, t, h]
        result = result_ * mask.dimshuffle(0, 1, 'x')  # [b, t, h]

        return result

