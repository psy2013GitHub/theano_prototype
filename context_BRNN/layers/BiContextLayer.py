#-*- encoding: utf8 -*-
__author__ = 'root'

from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from utils.theano_util import np_floatX
from keras.layers.core import Layer, MaskedLayer
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from keras import activations, initializations


class BiContextLayer(MaskedLayer):

    def __init__(self, output_dim, context_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='sigmoid', inner_activation='hard_sigmoid',
                 **kwargs):
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        super(BiContextLayer, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        input_dim = input_shape[2] # 嵌入维度
        self.e0 = self.init((input_dim,)) # 句子开头
        #print 'e0.type:', e0.type
        #self.e0 = e0.dimshuffle('x', 0, 1) # 样本维可广播
        #print 'self.e0.type:', self.e0.type
        self.c0 = self.init((self.context_dim,))
        #self.c0 = c0.dimshuffle('x', 0, 1)
        self.en = self.init((input_dim,)) # 句子结尾
        #self.en = en.dimshuffle('x', 0, 1)
        self.cn = self.init((self.context_dim,))
        #self.cn = cn.dimshuffle('x', 0, 1)

        self.Wl = self.init((self.context_dim, self.context_dim))
        self.Wr = self.init((self.context_dim, self.context_dim))
        self.Wsl = self.init((input_dim, self.context_dim))
        self.Wsr = self.init((input_dim, self.context_dim))
        self.W2 = self.init((input_dim + 2*self.context_dim, self.output_dim))
        self.b2 = shared_zeros((self.output_dim),)

        self.params = [self.e0, self.c0,
                       self.en, self.cn,
                       self.Wl, self.Wr,
                       self.Wsl, self.Wsr,
                       self.W2, self.b2]

    def get_output(self, train=False):
        X = self.get_input(train=train)
        c0 = self.c0[None,:] * T.ones((X.shape[0], self.context_dim))
        cn = self.cn[None,:] * T.ones((X.shape[0], self.context_dim))
        X = T.concatenate(
            [
                T.shape_padleft(self.e0,2) * T.ones((X.shape[0], 1, X.shape[2])),
                X,
                T.shape_padleft(self.en,2) * T.ones((X.shape[0], 1, X.shape[2])),
            ],
            axis = 1
        )
        X = X.dimshuffle(1,0,2) # timestep 置于第一纬
        # 只有将int32 mask 强制转换为 float32 才不会在scan里面将mask_t[:, None] * cl_t 结果upcast成float64
        mask = T.cast(self.get_output_mask(train=train), T.config.floatX)
        mask = mask.dimshuffle(1,0) # timestep 置于第一纬
        #theano.printing.debugprint([mask], print_type=True)
        def _forward_step(e_t, e_tm1, mask_t, cl_tm1):
            #print 'e_t:', e_t.type.ndim
            #print 'cl_t:', cl_tm1.type.ndim
            cl_t = T.nnet.sigmoid(
                T.dot(cl_tm1, self.Wl) + T.dot(e_tm1, self.Wsl)
            )
            cl_t = mask_t[:, None] * cl_t + (1. - mask_t[:, None]) * cl_tm1 # 如果它被mask就直接继承那个词
            #theano.printing.debugprint([mask_t], print_type=True)
            #theano.printing.debugprint([cl_t], print_type=True)
            return cl_t
        def _backward_step(e_t, e_tp1, mask_t, cr_tp1):
            cr_t = T.nnet.sigmoid(
            T.dot(cr_tp1, self.Wr) + T.dot(e_tp1, self.Wsr))
            cr_t = mask_t[:, None] * cr_t + (1. - mask_t[:, None]) * cr_tp1 # 如果它被mask就直接继承那个词
            return cr_t
        Cl, _ = theano.scan(_forward_step,
                        sequences=[dict(input=X, taps=[0, -1]), mask],
                        outputs_info=[
                            dict(initial=c0, taps=[-1]) # 注意不是c0!!!
                        ],

        )
        Cr, _ = theano.scan(_backward_step,
                            sequences=[dict(input=X, taps=[0, -1]), mask],
                            outputs_info=[
                                dict(initial=cn, taps=[-1])
                            ],
                            go_backwards=True,
        )
        Cr = Cr[::-1] # 翻转Cr
        def _concatenate_activation_step(e_t, mask_t, cl_t, cr_t):
            #print theano.printing.debugprint(cr_t, print_type=True)
            h_t = T.tanh( T.dot(T.concatenate([e_t, cl_t, cr_t], axis=1), self.W2)
                       + self.b2)
            h_t = mask_t[:, None] * h_t + (1. - mask_t[:, None]) * (-10000000000.) # 将mask的地方设置为最小值
            return h_t

        Y, _ = theano.scan(_concatenate_activation_step,
                    sequences=[X, mask, Cl, Cr],
                    outputs_info=None,
        )
        return Y.dimshuffle(1,0,2) # 重置样本为第一维

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[1], self.output_dim)

    def get_config(self):
        config = {
                    "name": self.__class__.__name__,
                    "output_dim": self.output_dim,
                    "init": self.init.__name__,
                 }
        base_config = super(BiContextLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))