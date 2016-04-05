#-*- encoding: utf8 -*-
from __future__ import division
__author__ = 'root'

import theano.tensor as T
from keras.layers.core import Layer, MaskedLayer

class Folding(MaskedLayer):
    '''
        折叠层, 即奇数与偶数相加平均
    '''

    input_ndim = 4 # n_sample, n_map, words_dim, embedding_dim
    def __init__(self):
        super(Folding, self).__init__()

    def get_output(self, train=False):
        self.get_folding_dim()
        X = self.get_input(train) # 样本 * 通道 * 维度
        return (X[:, :, :, T.arange(0, self.folding_dim, 2).astype('int32')] +
                X[:, :, :, T.arange(1, self.folding_dim, 2).astype('int32')]) / 2

    @property
    def output_shape(self):
        self.get_folding_dim()
        return  self.input_shape[:3] + (self.folding_dim / 2, )

    def get_folding_dim(self):
        self.folding_dim = self.input_shape[3]
        #assert self.folding_dim % 2 == 0, 'folding dim must be even, but now %d' % self.folding_dim
